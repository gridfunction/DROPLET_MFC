// make jko_lub && ./jko_lub  -tC 1 -tol 8 or ./jko_lub -tC 2 -tol 8
#include "mfem.hpp"
#include <fstream>
#include <iostream>
// optimization solver
#include "brent.hpp"

using namespace std;
using namespace mfem;

// CG parameters
double cgrtol=1e-6; 
double cgatol=1e-10; 
int cgi=100;
int iterPnt=10000; 
int gma = 0.01;// not used

int test_case = 1; 
double beta = 0.01;   // n = beta grad h
double alpha = 0.01;  // diffusion coefficient
int typeV1 = 3;       // mobility exponent
double eps = 0.3;     
int tolO = 8; // PDHG tolerance = 10^-tolO
// physical parameters
double c2 = 0.04, K = 0.1, P=0.5, xL = 1.0, yL = 1.0;
// initial spatial mesh size
int nx = 8, ny = 8;

bool per = true; // periodic BC
// PDHG parameters
double sigma_u = 1.0;   // primal step parameter
double sigma_phi = 1.0; // dual step parameter
const int double_bits = std::numeric_limits<double>::digits;

double rhoMin = 1e-2, rhoMax = 10.0;

double rho0(const Vector &);
double E(double);
double V1x(double);
double V2x(double);
double F(double rho, double rhobar, double mbar2,double sbar2, double tau);

class DiffusionMultigrid : public GeometricMultigrid
{
   std::unique_ptr<HypreBoomerAMG> amg;
   ConstantCoefficient rr; // mass matrix coeff.

public:
   // Constructs a diffusion multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries
   DiffusionMultigrid(ParFiniteElementSpaceHierarchy &hierarchy,
                      Array<int> &ess_bdr, double _rr) : GeometricMultigrid(hierarchy),rr(_rr)
   {
      ConstructCoarseOperatorAndSolver(hierarchy.GetFESpaceAtLevel(0), ess_bdr);
      for (int level = 1; level < hierarchy.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(hierarchy.GetFESpaceAtLevel(level), ess_bdr);
      }
   }

   void FormFineSystemMatrix(OperatorHandle &A)
   {
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), A);
   }

private:
   void ConstructBilinearForm(ParFiniteElementSpace &fespace,
                              Array<int> &ess_bdr,
                              bool partial_assembly)
   {
      ParBilinearForm* form = new ParBilinearForm(&fespace);
      if (partial_assembly)
      {
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      form->AddDomainIntegrator(new DiffusionIntegrator);
      form->AddDomainIntegrator(new MassIntegrator(rr));
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>);
      fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace &coarse_fespace,
                                         Array<int> &ess_bdr)
   {
      ConstructBilinearForm(coarse_fespace, ess_bdr, false);

      HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

      amg.reset(new HypreBoomerAMG(*hypreCoarseMat));
      amg->SetPrintLevel(-1);

      CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
      pcg->SetPrintLevel(-1);
      pcg->SetMaxIter(10);
      pcg->SetRelTol(1e-4);
      pcg->SetAbsTol(0.0);
      pcg->SetOperator(*hypreCoarseMat);
      pcg->SetPreconditioner(*amg);

      AddLevel(hypreCoarseMat, pcg, true, true);
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace,
                                     Array<int>& ess_bdr)
   {
      // FIXME: due to bdry term, no partial assembly
      ConstructBilinearForm(fespace, ess_bdr, false);

      HypreParMatrix *matrix = new HypreParMatrix;
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), *matrix);

      Vector diag(fespace.GetTrueVSize());
      bfs.Last()->AssembleDiagonal(diag);

      Solver* smoother = new OperatorChebyshevSmoother(
         *matrix, diag, *essentialTrueDofs.Last(), 2, fespace.GetParMesh()->GetComm());

      AddLevel(matrix, smoother, true, true);
   }
};

class MultigridHierarchy
{
   H1_FECollection coarse_fec;
   ParFiniteElementSpace coarse_space;
   vector<unique_ptr<H1_FECollection>> fe_collections;
   ParFiniteElementSpaceHierarchy space_hierarchy;
   const int order;
public:
   MultigridHierarchy(ParMesh &coarse_mesh, int h_refs, int p_refs)
   : coarse_fec(1, coarse_mesh.Dimension()),
     coarse_space(&coarse_mesh, &coarse_fec),
     space_hierarchy(&coarse_mesh, &coarse_space, false, false),
     order(pow(2, p_refs))
   {
      const int dim = coarse_mesh.Dimension();
      for (int level = 0; level < h_refs; ++level)
      {
         space_hierarchy.AddUniformlyRefinedLevel();
      }
      for (int level = 0; level < p_refs; ++level)
      {
         fe_collections.push_back(unique_ptr<H1_FECollection>(
            new H1_FECollection(pow(2, level+1), dim)
         ));
         space_hierarchy.AddOrderRefinedLevel(fe_collections.back().get());
      }
   }

   ParFiniteElementSpace &GetFinestSpace()
   {
      return space_hierarchy.GetFinestFESpace();
   }

   ParMesh &GetFinestMesh()
   {
      return *space_hierarchy.GetFinestFESpace().GetParMesh();
   }

   ParFiniteElementSpaceHierarchy &GetSpaceHierarchy() { return space_hierarchy; }

   int GetOrder() const { return order; }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/inline-quad.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 1;
   int geometric_refinements = 0;
   int order_refinements = 2;
   int order_red = 1;
   double  tau = 0.001;// time step size
   double  tend = 0.4;// time step size
   bool paraview = true;
   int iterALGMax = 100000;
   int saveit = 200;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&iterALGMax, "-alg", "--iterALG", "Total ALG iteration counts");
   args.AddOption(&saveit, "-si", "--saveit", "Save every # iterations");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
   args.AddOption(&order_red, "-od", "--order-reduced",
                  "Reduced order for L2 function/integration rule.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Save data files for ParaView visualization.");
   args.AddOption(&test_case, "-tC", "--tC", "test Case");
   args.AddOption(&tau, "-tau", "--tau", "time step size");
   args.AddOption(&tend, "-tend", "--tend", "tend");
   args.AddOption(&beta, "-beta", "--beta", "beta");
   args.AddOption(&alpha, "-alpha", "--alpha", "alpha");
   args.AddOption(&c2, "-c2", "--c2", "c2");
   args.AddOption(&typeV1, "-tV1", "--tV1", "tV1");
   args.AddOption(&P, "-P", "--P", "pressure");
   args.AddOption(&tolO, "-tol", "--tol", "tol");
   args.AddOption(&xL, "-xL", "--xL", "Domain x length");
   args.AddOption(&yL, "-yL", "--yL", "Domain y length");
   args.AddOption(&nx, "-nx", "--nx", "# of cells per x-direction");
   args.AddOption(&ny, "-ny", "--ny", "# of cells per y-direction");
   args.AddOption(&sigma_u, "-su", "--su", "PDHG parameter u");
   args.AddOption(&sigma_phi, "-sp", "--sp", "PDHG parameter phi");
   args.AddOption(&per, "-per", "--periodic", "-no-per", "--no-periodic",
                  "periodic flag");
   args.ParseCheck();
   
   // save data every nSave JKO steps 
   int nSave = int(0.01/tau+0.5);

   // 
   Mesh serial_mesh;
   if (test_case==1){
     serial_mesh = Mesh::MakeCartesian1D(nx, xL);
   }else if (test_case==2){
     serial_mesh = Mesh::MakeCartesian2D(nx, ny, 
         Element::QUADRILATERAL, false, xL, yL);
   }

   const int dim = serial_mesh.Dimension();
   if (per){// the periodic mesh
       if (dim==1){
         std::vector<int> v2v(serial_mesh.GetNV());
         for (int i = 0; i < serial_mesh.GetNV(); ++i)
            v2v[i] = i;
         // Modify the mapping so that the last vertex gets mapped to the first vertex.
         v2v.back() = 0;
         serial_mesh = Mesh::MakePeriodic(serial_mesh, v2v); // Create the periodic mesh
       }else if (dim==2){
         // Create translation vectors defining the periodicity
         Vector x_translation({xL, 0.0});
         Vector y_translation({0.0, yL});
         std::vector<Vector> translations = {x_translation, y_translation};
         // Create the periodic mesh using the vertex mapping defined by the translation vectors
         serial_mesh = Mesh::MakePeriodic(serial_mesh, serial_mesh.CreatePeriodicVertexMapping(translations));
       }
   }


   // serial mesh -> coarse parallel mesh
   ParMesh coarse_mesh = [&]()
   {
     for (int l = 0; l < ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
     ParMesh par_mesh(MPI_COMM_WORLD, serial_mesh);
     serial_mesh.Clear();
     for (int l = 0; l < par_ref_levels; l++) { par_mesh.UniformRefinement(); }
     return par_mesh;
   }();
   
   MultigridHierarchy mg_hierarchy(coarse_mesh, geometric_refinements, order_refinements);

   const int order = mg_hierarchy.GetOrder()-order_red; // L2 space order
   L2_FECollection fec_W(order, dim, BasisType::GaussLegendre);
   H1_FECollection fec_V(order+order_red, dim);
   RT_FECollection fec_V1(order, dim);

   ParMesh &mesh = mg_hierarchy.GetFinestMesh();
   ParFiniteElementSpace &Vh = mg_hierarchy.GetFinestSpace();
   ParFiniteElementSpace Wh(&mesh, &fec_W);
   ParFiniteElementSpace Vh1(&mesh, &fec_V1); 
   ParFiniteElementSpace Wh1(&mesh, &fec_W, dim); 

   HYPRE_BigInt nVG = Vh.GlobalTrueVSize();
   HYPRE_BigInt nWG = Wh.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: " << nVG << "," << nWG << endl;
   }

   // integration rule
   const IntegrationRule &ir = IntRules.Get(mesh.GetElementGeometry(0), 
       2*order +1);

   ParGridFunction phi_gf(&Vh);
   ParGridFunction phi0_gf(&Vh);
   ParGridFunction sigma_gf(&Vh1);
   ParGridFunction sigma0_gf(&Vh1);
   ParGridFunction sigma_l2(&Wh1);
   ParGridFunction dsigma_l2(&Wh);
   ParGridFunction rho_gf(&Wh);
   DivergenceGridFunctionCoefficient dsigma_coeff(&sigma0_gf);
   VectorGridFunctionCoefficient sigma_coeff(&sigma0_gf);

   // Create the QuadratureSpace and QuadratureFUnctions to manage quantities
   // defined at quadrature points.
   QuadratureSpace qs(mesh, ir);
   QuadratureFunction rho_qf(qs);       // rho
   QuadratureFunction rhoOld_qf(qs);    // rho
   QuadratureFunction rho0_qf(qs);      // rho
   QuadratureFunction m_qf(qs, dim);    // flux
   QuadratureFunction n_qf(qs, dim);    // beta * grad(rho)
   QuadratureFunction s_qf(qs);         // source
   
   QuadratureFunction phi0_qf(qs);            // incremental phi
   QuadratureFunction dphi0_qf(qs, dim);      // incremental phi
   QuadratureFunction sigma0_qf(qs, dim);     // incremental phi
   QuadratureFunction dsigma0_qf(qs, dim*dim); 
   
   QuadratureFunction drho_qf(qs);      // -phih0-beta*grad(sigmah0)
   QuadratureFunction dm_qf(qs, dim);   // grad(phih0)
   QuadratureFunction dn_qf(qs, dim);   // sigmah0
   QuadratureFunction ds_qf(qs);        // -sigma
   
   FunctionCoefficient(rho0).Project(rho0_qf);
   
   // initialization
   rho_qf = rho0_qf;
   m_qf = 0.0;
   n_qf = 0.0;
   s_qf=0.0;
   phi_gf=0.0;
   sigma_gf=0.0;
   
   // boundary
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   
   // Psi has boundary condition Psi.n=0
   Array<int> ess_tdof_list, ess_vdof_list, dofs_marker;
   if (!per){ 
     Array<int> ess_bdr1(mesh.bdr_attributes.Max()); 
     ess_bdr1 = 1; 
     Vh1.GetEssentialTrueDofs(ess_bdr1, ess_tdof_list);
     Vh1.GetEssentialVDofs(ess_bdr1, dofs_marker);
     FiniteElementSpace::MarkerToList(dofs_marker, ess_vdof_list);
   }

   int nV = Vh.GetTrueVSize();
   int nV1 = Vh1.GetTrueVSize();
   int nW = Wh.GetTrueVSize();
   OperatorPtr A, A1;
   Vector X(nV), B(nV), B1(nV1), X1(nV1);
   X=0.0; X1=0.0; 
  

   // Use Multigrid preconditioner for diffusion
   DiffusionMultigrid MG(mg_hierarchy.GetSpaceHierarchy(), ess_bdr, 2.0);
   MG.SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   
   MG.FormFineSystemMatrix(A);
   // 11. Solve the linear system A X = B.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*A);
   cg.SetPreconditioner(MG);
   cg.SetPrintLevel(-1);
   cg.SetMaxIter(cgi);
   cg.SetRelTol(cgrtol);
   cg.SetAbsTol(cgatol);
   
   ConstantCoefficient b2(beta*beta); // div-div coef.
   ParBilinearForm a1(&Vh1);

   a1.AddDomainIntegrator(new DivDivIntegrator(b2));
   a1.AddDomainIntegrator(new VectorFEMassIntegrator);
   a1.Assemble();
   a1.FormSystemMatrix(ess_tdof_list, A1);
   Solver *prec = new OperatorJacobiSmoother(a1, ess_tdof_list);

   CGSolver cg1(MPI_COMM_WORLD);
   cg1.SetOperator(*A1);
   cg1.SetPreconditioner(*prec);
   cg1.SetPrintLevel(-1);
   cg1.SetMaxIter(cgi);
   cg1.SetRelTol(cgrtol);
   cg1.SetAbsTol(cgatol);

   if (Mpi::Root()) { cout << "Done. " << tic_toc.RealTime() << endl; }

   // Set up the RHS. Will re-assemble every time.
   QuadratureFunction scalar_rhs_qf(qs);
   QuadratureFunction vector_rhs_qf(qs, dim);
   QuadratureFunctionCoefficient scalar_rhs_coeff(scalar_rhs_qf);
   VectorQuadratureFunctionCoefficient vector_rhs_coeff(vector_rhs_qf);
   
   auto set_rhs_qf = [&](double sigma_phi)
   {
      const int nq = ir.Size();
      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         for (int iq = 0; iq < nq; ++iq)
         {
            const int idx = iel*nq + iq;
            scalar_rhs_qf[idx] = (rho_qf[idx]-rho0_qf[idx]-s_qf[idx])*sigma_phi;
            for (int d = 0; d < dim; ++d)
            {
               const int d_idx = idx*dim + d;
               vector_rhs_qf[d_idx] = -m_qf[d_idx]*sigma_phi;
            }
         }
      }
   };
   
   auto set_rhs_qf1 = [&](double sigma_phi)
   {
      const int nq = ir.Size();
      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         for (int iq = 0; iq < nq; ++iq)
         {
            const int idx = iel*nq + iq;
            scalar_rhs_qf[idx] = beta * ( - rho_qf[idx] * sigma_phi + phi0_qf[idx]);
            for (int d = 0; d < dim; ++d)
            {
               const int d_idx = idx*dim + d;
               vector_rhs_qf[d_idx] = - n_qf[d_idx]*sigma_phi;
            }
         }
      }
   };
   
   ParLinearForm b(&Vh);
   b.AddDomainIntegrator(new DomainLFIntegrator(scalar_rhs_coeff));
   b.AddDomainIntegrator(new DomainLFGradIntegrator(vector_rhs_coeff));
   
   ParLinearForm b1(&Vh1);
   b1.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vector_rhs_coeff));
   b1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(scalar_rhs_coeff));
   
   
   // Tell the linear form to use the specific quadrature rule
   for (auto *integ : *b.GetDLFI())
        integ->SetIntRule(&ir);
   for (auto *integ : *b1.GetDLFI())
        integ->SetIntRule(&ir);
   b.UseFastAssembly(true);
   b1.UseFastAssembly(true);

   enum EvalMode
   {
      QF_VALS = 0,
      QF_GRAD = 1,
      QF_DIV = 2
   };

   Vector e_vec;
   auto qpt_interp = [&](const GridFunction &gf, 
       QuadratureFunction &qf, EvalMode mode)
   {
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      auto fes = gf.FESpace();
      const Operator *R = fes->GetElementRestriction(ordering);
      e_vec.SetSize(R->Height());
      R->Mult(gf, e_vec);
      // Use quadrature interpolator to go from E-vector to Q-vector
      const QuadratureInterpolator *qi = fes->GetQuadratureInterpolator(qs);
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      qi->DisableTensorProducts(false);

      if (mode == QF_VALS) { qi->Values(e_vec, qf); }
      else if (mode == QF_GRAD) { qi->PhysDerivatives(e_vec, qf); }
   };
   
   StopWatch sw_a, sw_b, sw_c, sw_d, sw_e;
   double rt[5];

   int iterALG = 0;
   
   iterALG = 0;
   sw_a.Clear();
   sw_b.Clear();
   sw_c.Clear();
   sw_d.Clear();
   sw_e.Clear();
   
   int ip = 0;
   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      string file = "JKO_D"+to_string(dim)+"T"+to_string(tolO);
      // save data
      for (int j = 0; j < nW; j++){
          rho_gf(j) = rho_qf(j);
      }
      if (Mpi::Root())
          cout << "save to ..."<< file << endl;
      pd = new ParaViewDataCollection(file, &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("height", &rho_gf);
      pd->RegisterField("phi", &phi_gf);
      pd->SetLevelsOfDetail(order+order_red);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(ip);
      pd->SetTime(ip++);
      pd->Save();
   }
   
   // JKO loop
   double time = 0.0, err = 1.0;
   int iterJKO = 0, iter=0;
   // save errors 
   std::vector<int> njkoList;
   
   int sizeR = rho_qf.Size();
   MPI_Allreduce(MPI_IN_PLACE, &sizeR, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   while (time < tend){
     time += tau;
     iterJKO++;
     err = 1.0;
     iterALG = 0;
     // Optimization loop 
     while (iterALG < iterALGMax)
     {
        // PHI: Assembly
        set_rhs_qf(sigma_phi);
        b.Assemble(); 
        b.ParallelAssemble(B);
        X = 0.0;
        sw_a.Start();
        cg.Mult(B, X);
        sw_a.Stop();
        // get phi0_qf
        phi0_gf.SetFromTrueDofs(X);
        qpt_interp(phi0_gf, phi0_qf, QF_VALS);
        rt[0] = sw_a.RealTime();

        // Sigma: Assembly
        set_rhs_qf1(sigma_phi);
        b1.Assemble(); 
        b1.ParallelAssemble(B1);
        X1 = 0.0;
        sw_b.Start();
        cg1.Mult(B1, X1);
        sw_b.Stop();
        rt[1] = sw_b.RealTime();
        sigma0_gf.SetFromTrueDofs(X1);
        if (!per)
            for (auto dd: ess_vdof_list)
              sigma0_gf[dd] = 0;

        sw_c.Start();
        // Update phiNew 
        phi_gf += phi0_gf;
        sigma_gf += sigma0_gf;
        // Update 2*phiNew - phiOld = 2*delPhi + phiOld
        phi0_gf += phi_gf;
        sigma0_gf += sigma_gf;
         
        // Interpolation
        qpt_interp(phi0_gf, phi0_qf, QF_VALS);
        qpt_interp(phi0_gf, dphi0_qf, QF_GRAD);
        // get divergence && values
        dsigma_l2.ProjectCoefficient(dsigma_coeff);
        sigma_l2.ProjectCoefficient(sigma_coeff);
        // ADD SIGMA contribution
        for (int j=0; j < nW; j++){
          drho_qf[j] = - phi0_qf[j] + beta*dsigma_l2[j];
          for (int d=0; d<dim; d++){
            int idx = j*dim+d; 
            int idx0 = d*nW+j;
            dn_qf[idx] = sigma_l2[idx0];
          }
        }
        dm_qf = dphi0_qf;
        ds_qf = phi0_qf;

        sw_c.Stop();
        rt[2] = sw_c.RealTime();
        
        sw_d.Start();
        // 13. nonlinear solver for rho_gf
        double rho, rhobar, mbar2, sbar, sbar2;
        Vector mbar(dim), nbar(dim);
        std::pair<double, int> res;
        iter=0;
        // store old data
        rhoOld_qf = rho_qf;
        
        for (int j = 0; j < nW; j++)
        {
           mbar2 = 0;
           for (int d=0; d<dim;d++)
           {
              int idx = j*dim+d;
              mbar(d) = sigma_u * dm_qf[idx] + m_qf[idx];
              nbar(d) = sigma_u * dn_qf[idx] + n_qf[idx];
              mbar2 += pow(mbar(d), 2);
           }
           rhobar = sigma_u * drho_qf[j] + rho_qf[j];
           rho = rho_qf[j]; // initial guess 
           sbar = sigma_u * ds_qf[j] + s_qf[j];
           sbar2 = sbar*sbar;
           
           // Brent solver
           std::uintmax_t it = 20;
           auto func = [rhobar, mbar2, sbar2, tau](double x) { return 
               F(x, rhobar, mbar2, sbar2, tau); };
           auto res = boost::math::tools::brent_find_minima(func, rhoMin, rhoMax,
               double_bits, it);
           rho = res.first;
           if (it > iter) iter = it;

           for (int d = 0; d < dim; d++)
           {
              int idx = j*dim+d;
              m_qf[idx] = V1x(rho)/(sigma_u+V1x(rho))*mbar(d);
              n_qf[idx] = 1.0/(sigma_u*tau*pow(alpha/beta,2)+1)*nbar(d);
           }
           rho_qf[j] = rho;
           s_qf[j] = V2x(rho)/(sigma_u+V2x(rho))*sbar;
        }
        sw_d.Stop();
        rt[3] = sw_d.RealTime();

        iterALG++;
        // update error
        rhoOld_qf -= rho_qf;
        
        // compute L1 error
        err = rhoOld_qf.Norml1();
        MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        err /= sizeR;
        
        if (myid==0 and iterALG%iterPnt==0){
             cout << std::scientific ;
             cout << "Time " << std::setprecision(3)<< std::setw(2) << time/xL/xL
                << " JKO step " << std::setw(2) << iterJKO
                << " ALG step " << std::setw(2) << iterALG
                << " tau " << std::setw(2) << tau
                << " Brent " << std::setw(3) << iter
                << ",  " <<"CG: "<< cg.GetNumIterations()
                << ",  "<< "CG1: "<< cg1.GetNumIterations()
                << " ,  "<<"err: "<< err
                << " ,  "<<"PHI: "<< rt[0]
                << " ,  "<<"SIGMA: "<< rt[1]
                << " ,  "<<"U: "<< rt[3]
                << endl;
         }
          
        // early exit
        if (err < pow(10, -tolO))
            break;
     }
     njkoList.push_back(iterALG);
     sw_a.Clear();
     sw_b.Clear();
     sw_c.Clear();
     sw_d.Clear();
     double rhomin = rho_qf.Min();
     double rhomax = rho_qf.Max();
     MPI_Allreduce(MPI_IN_PLACE, &rhomin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
     MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
     MPI_Allreduce(MPI_IN_PLACE, &iter, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
     // printing JKO steps
     if (myid==0 and iterJKO%nSave==0 ){
         cout << std::scientific ;
         cout << "Time " << std::setprecision(3)<< std::setw(2) << time/xL/xL
            << " JKO step " << std::setw(2) << iterJKO
            << " ALG step " << std::setw(2) << iterALG
            << " tau " << std::setw(2) << tau/xL/xL
            << " Brent " << std::setw(3) << iter
            << ",  " <<"CG: "<< cg.GetNumIterations()
            << ",  "<< "CG1: "<< cg1.GetNumIterations()
            << " ,  "<<"err: "<< err
            << " ,  "<<"PHI: "<< rt[0]
            << " ,  "<<"SIGMA: "<< rt[1]
            << " ,  "<<"U: "<< rt[3]
            << endl;
     }
     
     // update density
     rho0_qf = rho_qf;
     // Save data
     if (paraview && iterJKO%nSave==0){
         // save data
         for (int j = 0; j < nW; j++){
             rho_gf(j) = rho_qf(j);
         }
         pd->SetCycle(ip++);
         pd->SetTime(time);
         pd->Save();
     }

   }
   
   if (Mpi::Root()){
       // save PDHG converge history
       std::ofstream outputFile("ParaView/D"+to_string(dim)+"_njkoT"+to_string(tolO)+".txt");
       // Write vector elements to the file
       for (auto value : njkoList) {
           outputFile << value << endl;
       }
       // Close the file
       outputFile.close();
   }

   return 0;
}

// initial density (scaled with -1/r)
double rho0(const Vector &x)
{
   double val=0.0;
   if (test_case==1)
     val = 1.0-0.2*cos(2*M_PI*x(0)/xL);
   else if (test_case==2)
     val = 1.0+0.2*cos(2*M_PI*x(0)/xL)*cos(2*M_PI*x(1)/xL);
   return val;
}

// interaction potential
double E(double rho){
   double h = eps/rho;
   double val = pow(h, 3)/3.0 - pow(h, 2)/2.0-rho*P;
   return val;
}

// reaction rate
double V1x(double rho){
   double val;
   val = pow(rho, typeV1);
   return val;
}

// reaction rate
double V2x(double rho){
   double val;
   val = c2/(rho+K);
   return val;
}
    
double F(double rho, double rhobar, double mbar2,double sbar2, double tau){
    double val;
    val = pow(rho-rhobar,2)/sigma_u+2*tau*E(rho) + mbar2/(sigma_u+V1x(rho))+ sbar2/(sigma_u+V2x(rho));
    return val;
}
