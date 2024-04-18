// st-ho-JKO for droplet dynamics
// testcase=12: 1D case
// testcase=13: 2D case
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "kron_quad.hpp"

#include "divdiv.hpp"
#include "wass_laplace.hpp"
#include "wass_multigrid.hpp"
#include "wass_rhs.hpp"
#include "wass_params.hpp"

// optimization solver
#include "brent.hpp"

using namespace std;
using namespace mfem;

double eps = 0.3; // layer heights
double cgrtol = 1e-6; // cg relative tolerance
double cgatol = 1e-10; // cg absolute tolerance
int cgi = 1000; // max # of cg iterations
int iterPnt = 100; // print every iterPnt steps
double gma = 0.01, beta = 0.01; // FIXME: we always take gma==beta
int typeI = 0, typeV1 = 3, typeV2 = 3; 
double typeE = 1.0;  // typeE = 1: E = rho(log(rho)-1)
                     // typeE > 1: E = rho^typeE/(typeE-1)
double c2 = 0.04; // reaction strength
double sigma_phi = 1.0, sigma_u = 1.0; // PDHG parameters: always set to 1
int test_case = 32; // 1D results

bool jac = true;// jacobi smoother for div-div
// Brent solver end ponts
double rmin = 1e-4, rmax = 10.0;
double t_end = 0.01;
bool ho = true; // FIXME: high-order flag

// control parameters
double ci = 0.04, cb = 0.05, cB = 0;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "data/square-quad.mesh";
   int n_time = 4;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int h_refs = 4;
   int p_refs = 0;
   int maxit = 5000;

   OptionsParser args(argc, argv);
   // Mesh parameters & time step size
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&n_time, "-nt", "--n-coarse-time", "Number of coarse time elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", "Serial refinements.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel", "Parallel refinements.");
   args.AddOption(&h_refs, "-hr", "--h-refs", "Mesh (h) multigrid refinements.");
   args.AddOption(&p_refs, "-pr", "--p-refs", "Degree (p) multigrid refinements.");
   args.AddOption(&test_case, "-tc", "--test-case", "Test case: 1, 2, or 3");
   // physical & control parameters
   args.AddOption(&gma, "-gamma", "--gamma", "gamma");
   args.AddOption(&eps, "-eps", "--eps", "eps");
   args.AddOption(&beta, "-beta", "--beta", "beta");
   args.AddOption(&t_end, "-tend", "--tend", "final time");
   args.AddOption(&cb, "-cb", "--cb", "KL terminal strength");
   //args.AddOption(&cB, "-cB", "--cB", "obstacle strength");
   args.AddOption(&ci, "-ci", "--ci", "interaction potential strength");
   args.AddOption(&c2, "-c2", "--c2", "reaction strength");
   // Solver parameters
   args.AddOption(&maxit, "-alg", "--alg", "# alg iterations");
   args.AddOption(&cgrtol, "-cgr", "--cgrtol", "cg relative tolerance");
   args.AddOption(&cgatol, "-cga", "--cgatol", "cg absolute tolerance");
   args.AddOption(&cgi, "-cgi", "--cgi", "cg max iteration");
   args.AddOption(&iterPnt, "-iP", "--iterPnt", "print every # steps");
   args.AddOption(&jac, "-jac", "--use-jac", "-lor", "--use-lor", "Use LOR or JAC preconditioner");
   args.AddOption(&ho, "-ho", "--use-ho", "-lo", "--use-lo", "Use HO");
   // Parse arguments
   args.ParseCheck();

   // FIXME: const time stepping
   const double Tf = 1.0;
   const int maxJKO = (beta==0)? 1 : t_end/beta;
      
   string fileX = 'T'+to_string(test_case)+'H'+to_string(h_refs)+'P'+to_string(p_refs) \
                   +'B'+to_string(int(cb*100+0.1))+'I'+to_string(int(ci*100+0.1));
   //                +'O'+to_string(int(cB+0.1));

   ParMesh coarse_space_mesh = [&]()
   {
      Mesh serial_mesh;
      if (test_case == 36) // 2D with complex geom
          serial_mesh = Mesh::LoadFromFile(mesh_file);
      else if (test_case/10 == 2) // 1D drop tests
         serial_mesh = Mesh::MakeCartesian1D(16, 1.0);
      else if (test_case/10 == 3) // 2D drop tests
         serial_mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, 
             false, 1.0, 1.0);
      else
         serial_mesh = Mesh::LoadFromFile(mesh_file);

      for (int l = 0; l < ser_ref_levels; l++) { serial_mesh.UniformRefinement(); }
      ParMesh par_mesh(MPI_COMM_WORLD, serial_mesh);
      serial_mesh.Clear();
      for (int l = 0; l < par_ref_levels; l++) { par_mesh.UniformRefinement(); }
      return par_mesh;
   }();

   ParMesh coarse_time_mesh = [&]()
   {
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Group local_group;
      int ranks[1] = { Mpi::WorldRank() };
      MPI_Group_incl(world_group, 1, ranks, &local_group);
      MPI_Comm local_comm;
      MPI_Comm_create(MPI_COMM_WORLD, local_group, &local_comm);
      Mesh serial_mesh = Mesh::MakeCartesian1D(n_time, Tf);
      return ParMesh(local_comm, serial_mesh);
   }();

   MultigridHierarchy space_hierarchy(coarse_space_mesh, h_refs, p_refs);
   MultigridHierarchy time_hierarchy(coarse_time_mesh, h_refs, p_refs);

   const int order_h1 = space_hierarchy.GetOrder();
   const int order_l2 = order_h1 - 1;
   ParMesh &space_mesh = space_hierarchy.GetFinestMesh();
   ParMesh &time_mesh = time_hierarchy.GetFinestMesh();

   const int dim_s = space_mesh.Dimension(); // space dimension
   const int dim = dim_s + 1; // spacetime dimension

   ParFiniteElementSpace &Vs = space_hierarchy.GetFinestSpace();
   ParFiniteElementSpace &Vt = time_hierarchy.GetFinestSpace();

   L2_FECollection l2_fec(order_h1 - 1, 1);
   ParFiniteElementSpace St(&time_mesh, &l2_fec);

   RT_FECollection rt_fec(order_h1 - 1, dim_s, BasisType::GaussLobatto, BasisType::IntegratedGLL);
   ParFiniteElementSpace Ss(&space_mesh, &rt_fec, 1);

   Array<int> ess_bdr_s(space_mesh.bdr_attributes.Max());
   ess_bdr_s = 1;
   Array<int> ess_dofs_s, ess_dofs_v;
   Ss.GetEssentialTrueDofs(ess_bdr_s, ess_dofs_s);
   Vs.GetEssentialTrueDofs(ess_bdr_s, ess_dofs_v);

   const int nVs = Vs.GetTrueVSize();
   const int nVt = Vt.GetTrueVSize();
   const int nV = nVs*nVt; // scalar phi

   const int nSs = Ss.GetTrueVSize();
   const int nSt = St.GetTrueVSize();
   const int nS = nSs*nSt; // vector sigma & theta
   
   const int nX = nVs*nSt; // scalar xi

   {
      const HYPRE_BigInt nVs_global = Vs.GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "Number of space unknowns:  " << nVs_global << '\n'
              << "Number of time unknowns:   " << nVt << '\n'
              << "Total number of unknowns:  " << nVs_global*nVt << '\n'
              << "H1 order: " << order_h1 << '\n'
              << "RT_" << order_h1-1 << '\n'
              << endl;
      }
   }

   // Do some sanity checks to make sure we correctly index into the temporal
   // vectors. This is needed to put the initial and terminal boundary terms in
   // the right place in the space-time vectors.
   {
      GridFunction time_nodes(&Vt); // time coordinates
      time_mesh.GetNodes(time_nodes);
      // Make sure the slice corresponding to t=0 is at offset 0...
      MFEM_VERIFY(time_nodes[0] == 0.0, "");
      // Make sure the slice corresponding to dt is at offset n_time...
      MFEM_VERIFY(time_nodes[n_time] == Tf, "");
   }

   const Geometry::Type geom_s = space_mesh.GetElementGeometry(0);
   const Geometry::Type geom_t = time_mesh.GetElementGeometry(0);
   const int q_gl = 2*order_l2 + 1;

   const IntegrationRule &ir_s = IntRules.Get(geom_s, q_gl);
   const IntegrationRule &ir_t = IntRules.Get(geom_t, q_gl);

   QuadratureSpace Qs(space_mesh, ir_s);
   QuadratureSpace Qt(time_mesh, ir_t);
   const int nQs = Qs.GetSize();
   const int nQt = Qt.GetSize();
   const int nQ = nQs*nQt;

   // Set up coefficients and initial conditions
   FunctionCoefficient rho_drop_ex_coeff(rho_drop_ex);
   FunctionCoefficient rho_obs_coeff(rho_obs);
   FunctionCoefficient rho_drop_target_coeff(rho_drop_target);
   
   // terminal data rho_T & n_T
   QuadratureFunction rho_T_qf(Qs);
   QuadratureFunction rho_T1_qf(Qs); // target density
   // initialize
   FunctionCoefficient([Tf](const Vector &xvec){
      return rho_drop_ex(xvec, Tf);
   }).Project(rho_T_qf);
   rho_drop_target_coeff.Project(rho_T1_qf);

   QuadratureFunctionCoefficient rho_T_coeff(rho_T_qf);
   QuadratureFunction n_T_qf(Qs, dim_s);
   
   // Set by evaluating space-time grid function at the terminal time value
   QuadratureFunction dphi_T_qf(Qs);
   KroneckerFaceInterpolator face_interp(Vs, Qs);

   // Physical variables
   // rho : vdim = 1 (scalar)
   // m   : vdim = dim_s
   // n   : vdim = dim_s
   // s   : vdim = 1 (scalar)
   // p   : vdim = 1 (scalar)
   // q   : vdim = dim_s
   const int u_vdim = 1 + dim_s + dim_s;
   const int pq_vdim = 1 + dim_s;

   KroneckerQuadratureFunction u_qf(Qs, Qt, u_vdim); // (rho, m, n)
   KroneckerQuadratureFunction pq_qf(Qs, Qt, pq_vdim); // (p, q)
   KroneckerQuadratureFunction s_qf(Qs, Qt, u_vdim); // (rho, m, n)
   
   // FIXME: no obstacle for now :(
   //KroneckerQuadratureFunction rhoB_qf(Qs, Qt, 1); // obstacle
   //rhoB_qf.ProjectComponent(rho_obs_coeff, 0);

   u_qf = 0.0;
   u_qf.ProjectComponent(rho_drop_ex_coeff, 0);
   pq_qf = 0.0;
   s_qf = 0.0;
   n_T_qf = 0.0;

   KroneckerQuadratureFunction dphi_qf(Qs, Qt, dim); // grad(phi)
   KroneckerQuadratureFunction ds_qf(Qs, Qt, 1);    // phi
   KroneckerQuadratureFunction dsigma_qf(Qs, Qt, dim_s); // sigma
   KroneckerQuadratureFunction divs_qf(Qs, Qt, 1); // div(sigma)
   KroneckerQuadratureFunction dxi_qf(Qs, Qt, 1);  // xi
   KroneckerQuadratureFunction gradxi_qf(Qs, Qt, dim_s);  // xi
   KroneckerQuadratureFunction dtheta_qf(Qs, Qt, dim_s);  // theta
   KroneckerQuadratureFunction divt_qf(Qs, Qt, 1); // div(theta)
   
   QuadratureFunction dsigma_T_qf(Qs, dim_s); // sigmaT
   QuadratureFunction divs_T_qf(Qs, 1); // div(sigmaT)

   KroneckerQuadratureFunction rho_err_qf(Qs, Qt, 1); // aux vector for error calculation
   QuadratureFunction rho_T_err_qf(Qs, 1); // aux vector for terminal error calculation

   // Right-hand side for phi
   KroneckerLinearForm B_0(Vs, Vt, Qs, Qt);
   // Right-hand side for sigma
   KroneckerFisherLinearForm B_1(Ss, St, Qs, Qt);
   // Right-hand side for xi
   KroneckerXiLinearForm B_2(Vs, St, Qs, Qt);
   // Right-hand side for theta
   KroneckerFisherLinearForm B_3(Ss, St, Qs, Qt);
   // Right-hand side for sigma_T
   DivTLinearForm B_T(Ss, Qs);
   
   // Boundary terms
   Vector B_bdr_0(nVs);
   Vector B_bdr_T(nVs);
   // Bottom boundary (t == 0)
   {
      FunctionCoefficient minus_rho_0_coeff([](const Vector &xvec) { return -rho_drop_ex(xvec, 0.0); });
      ParLinearForm b_bdr_0(&Vs);
      b_bdr_0.AddDomainIntegrator(new DomainLFIntegrator(minus_rho_0_coeff));
      for (auto *i : *b_bdr_0.GetDLFI()) { i->SetIntRule(&ir_s); }
      b_bdr_0.UseFastAssembly(true);
      b_bdr_0.Assemble();
      b_bdr_0.ParallelAssemble(B_bdr_0);
   }
   
   // Top boundary (t = 1)
   // We retain the ParLinearForm object, since we need to re-assembly every
   // iteration.
   ParLinearForm b_bdr_T(&Vs);
   b_bdr_T.AddDomainIntegrator(new DomainLFIntegrator(rho_T_coeff));
   for (auto *i : *b_bdr_T.GetDLFI()) { i->SetIntRule(&ir_s); }
   b_bdr_T.UseFastAssembly(true);
   b_bdr_T.Assemble();
   b_bdr_T.ParallelAssemble(B_bdr_T);
   
   // Set up the phi operator
   const double mass_coeff = 1.0;
   KroneckerMultigrid mg(space_hierarchy.GetSpaceHierarchy(),
                         time_hierarchy.GetSpaceHierarchy(),
                         mass_coeff,
                         n_time);

   KroneckerLaplacian A_0(Vs, Vt, mass_coeff, n_time);
   MFEM_VERIFY(nV == A_0.Height(), "Incompatible sizes.");

   CGSolver cg_0(MPI_COMM_WORLD);
   cg_0.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg_0.SetOperator(A_0);
   cg_0.SetPreconditioner(mg);
   cg_0.SetMaxIter(cgi);
   cg_0.SetRelTol(cgrtol);
   cg_0.SetAbsTol(cgatol);
   
   // Set up the solver for the xi operator
   KroneckerSpaceLaplacian A_2(Vs, St, ess_bdr_s);
   // TODO preconditioner for A_2
   CGSolver cg_2(MPI_COMM_WORLD);
   cg_2.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg_2.SetOperator(A_2);
   //cg_2.SetPreconditioner();
   cg_2.SetMaxIter(cgi);
   cg_2.SetRelTol(cgrtol);
   cg_2.SetAbsTol(cgatol);
   
   // Set up the solver for the sigma & theta operator
   KroneckerDivDivSolver div_div_solver(Ss, St, ess_bdr_s, jac);
   // Set up the solver for the sigma_T operator
   ParBilinearForm Ds(&Ss);
   OperatorHandle Ds_op;
   ConstantCoefficient Ds_coeff(gma*gma);
   Ds.AddDomainIntegrator(new DivDivIntegrator(Ds_coeff));
   Ds.AddDomainIntegrator(new VectorFEMassIntegrator);
   Ds.Assemble();
   Ds.FormSystemMatrix(ess_dofs_s, Ds_op);
   
   std::unique_ptr<Solver> prec_T;
   prec_T.reset(new OperatorJacobiSmoother(Ds, ess_dofs_s));
   
   CGSolver cg_T(MPI_COMM_WORLD);
   cg_T.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg_T.SetMaxIter(cgi);
   cg_T.SetRelTol(cgrtol);
   cg_T.SetAbsTol(cgatol);
   cg_T.SetOperator(*Ds_op);
   cg_T.SetPreconditioner(*prec_T);


   Vector X_0(nV); // phi
   Vector dX_0(nV); // phi increment
   X_0 = dX_0 = 0.0;
   
   Vector X_1(nS); // sigma
   Vector dX_1(nS); // sigma increment
   X_1 = dX_1 = 0.0;
   
   Vector X_2(nX); // xi
   Vector dX_2(nX); // xi increment
   X_2 = dX_2 = 0.0;
   
   Vector X_3(nS); // theta
   Vector dX_3(nS); // theta increment
   X_3 = dX_3 = 0.0;
   
   Vector X_T(nSs); // sigma_T
   Vector dX_T(nSs); // sigma_T increment
   X_T = dX_T = 0.0;
   ParGridFunction sigma_T_gf(&Ss);


   if (Mpi::Root())
   {
      std::cout << "\nJKO  PHDG  CG1   CG2   CG3   CG4   CG5   Nonlin Error  Error-T       "
                << "Phi time    Sigma time   Nonlin time\n";
      std::cout << std::string(100, '=') << std::endl;
   }


   // sigmaT-interpolators
   enum EvalMode
   {
      QF_VALS = 0,
      QF_DIVS = 1
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
      const QuadratureInterpolator *qi = fes->GetQuadratureInterpolator(Qs);
      qi->SetOutputLayout(QVectorLayout::byVDIM);
      qi->DisableTensorProducts(false);

      if (mode == QF_VALS) { qi->Values(e_vec, qf); }
      else if (mode == QF_DIVS) { qi->PhysDivergence(e_vec, qf); }
   };
   
   // Post-process
   auto vis = [&](bool lor, int kk)
   {
      ParMesh lor_space_mesh;
      if (lor)
      {
         lor_space_mesh = ParMesh::MakeRefined(space_mesh, order_h1, Quadrature1D::GaussLobatto);
      }
      ParMesh &vis_mesh = lor ? lor_space_mesh : space_mesh;
      const int vis_order = lor ? 0 : order_l2;

      L2_FECollection l2_space_fec(vis_order, dim_s, BasisType::GaussLegendre);
      ParFiniteElementSpace Ws(&vis_mesh, &l2_space_fec);

      L2_FECollection l2_time_fec(order_l2, 1, BasisType::GaussLegendre);
      ParFiniteElementSpace Wt(&time_mesh, &l2_time_fec);

      ParFiniteElementSpace WVs(&vis_mesh, &l2_space_fec, dim_s);

      const int nWs = Ws.GetTrueVSize();
      const int nWt = Wt.GetTrueVSize();

      GridFunction time_nodes(&Wt); // time coordinates
      time_mesh.GetNodes(time_nodes);

      ParGridFunction rho_gf(&Ws);
      ParGridFunction rho_err_gf(&Ws);
      ParGridFunction vel_gf(&WVs);
      ParGridFunction mom_gf0(&Ws);

      Vector tdof_vec(nWs);
      
      ParaViewDataCollection pv(lor ? "DROP_LOR_"+fileX+"_"+to_string(kk) : 
          "DROP_"+fileX+"_"+to_string(kk), &vis_mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(!lor);
      pv.SetLevelsOfDetail(vis_order + 1);
      pv.RegisterField("rho", &rho_gf);
      pv.RegisterField("rho_err", &rho_err_gf);
      pv.RegisterField("vel", &vel_gf);

      auto save_time = [&](const double t)
      {
         GetTimeSlice(u_qf, tdof_vec, t, 0, u_vdim, Ws, Wt, lor);
         rho_gf.SetFromTrueDofs(tdof_vec);
         
         // The error vector
         GetTimeSlice(rho_err_qf, tdof_vec, t, 0, 1, Ws, Wt, lor);
         rho_err_gf.SetFromTrueDofs(tdof_vec);

         for (int d=0; d<dim_s; d++)
         {
            GetTimeSlice(u_qf, tdof_vec, t, d+1, u_vdim, Ws, Wt, lor);
            mom_gf0.SetFromTrueDofs(tdof_vec);
            for (int idx =0; idx< nWs; idx++)
            {
               double val =  mom_gf0[idx]/rho_gf[idx];
               vel_gf[idx + d*nWs] = (abs(val) < 10) ? val : 0.0;
            }
         }

         pv.SetTime(t);
         pv.SetCycle(pv.GetCycle() + 1);
         pv.Save();
      };
      
      if (ho==true){
          for (int k = 0; k < 11; ++k) 
              save_time(0.1*k);
      }
      // terminal density
      rho_gf = rho_T_qf;
      rho_err_gf = rho_T_err_qf;
      pv.SetTime(1.1);
      pv.SetCycle(pv.GetCycle() + 1);
      pv.Save();
   };
   
   StopWatch sw_a, sw_b, sw_c, sw_d, sw_e, sw_f;
   

   // save errors 
   std::vector<double> errorList;
   std::vector<double> errorTList;
            
   // initial data, save
   vis(false, 0);         
   vis(true, 0);

   // JKO Loop
   for (int jko=0; jko < maxJKO ; jko++)
   {
       // ALG loop
       for (int iter = 0; iter < maxit; ++iter)
       {
          // (1) Solve for dphi
          sw_a.Start();
          // the (dynamic) linear form for phi
          B_0.Update(u_qf, s_qf);
          // Add the boundary contributions
          {
             Vector B_bdr_slice_0(B_0, 0, nVs);
             B_bdr_slice_0 += B_bdr_0;

             // Terminal boundary needs to be recomputed
             b_bdr_T.Assemble();
             b_bdr_T.ParallelAssemble(B_bdr_T);
             Vector B_bdr_slice_T(B_0, n_time*nVs, nVs);
             B_bdr_slice_T += B_bdr_T;
          }

          // Solve the system, dX_0 gives the t-dof values of dphi increment
          dX_0 = 0.0;
          cg_0.Mult(B_0, dX_0);
          // scale dX_0 with sigma_phi
          dX_0 *= sigma_phi;
          sw_a.Stop();

          // NOTE: dphi_qf is the whole space-time gradient of dX_0
          dphi_qf.ProjectGradient(dX_0, Vs, Vt);
          // Boundary interpolation (terminal time)
          face_interp.Project(dX_0, dphi_T_qf, n_time);

          X_0 += dX_0;
          dX_0 += X_0;

          if (ho){
              // use sigma/xi/theta to recover h.o. time 
              // (2) Solve for sigma
              sw_b.Start();
              // Fisher
              B_1.Update(u_qf, dphi_qf);
              // Enforce essential BCs
              for (int it = 0; it < nSt; ++it)
              {
                 Vector B_1_slice(B_1, it*nSs, nSs);
                 B_1_slice.SetSubVector(ess_dofs_s, 0.0);
              }
              dX_1 = 0.0;
              div_div_solver.Mult(B_1, dX_1);
              // scale dX_1 with sigma_phi
              dX_1 *= sigma_phi;
              sw_b.Stop();

              // NOTE: dsigma_qf is sigma function evaluation
              dsigma_qf.ProjectValue(dX_1, Ss, St);

              X_1 += dX_1;
              dX_1 += X_1;
              
              // (3) Solve for xi
              sw_c.Start();
              // Fisher
              B_2.Update(u_qf, pq_qf, dsigma_qf);
              for (int it = 0; it < nSt; ++it)
              {
                 Vector B_2_slice(B_2, it*nVs, nVs);
                 B_2_slice.SetSubVector(ess_dofs_v, 0.0);
              }
              dX_2 = 0.0;
              cg_2.Mult(B_2, dX_2);
              // scale dX_1 with sigma_phi
              dX_2 *= sigma_phi;
              sw_c.Stop();
              
              // NOTE: dsigma_qf is sigma function evaluation
              dxi_qf.ProjectValue(dX_2, Vs, St);

              X_2 += dX_2;
              dX_2 += X_2;

              // (4) Solve for theta
              sw_d.Start();
              B_3.Update(pq_qf, dxi_qf, true);
              // Enforce essential BCs
              for (int it = 0; it < nSt; ++it)
              {
                 Vector B_3_slice(B_3, it*nSs, nSs);
                 B_3_slice.SetSubVector(ess_dofs_s, 0.0);
              }
              dX_3 = 0.0;
              div_div_solver.Mult(B_3, dX_3);
              // scale dX_1 with sigma_phi
              dX_3 *= sigma_phi;
              sw_d.Stop();

              X_3 += dX_3;
              dX_3 += X_3;
          }
          // (5) Solve for sigma_T
          B_T.Update(rho_T_qf, n_T_qf, dphi_T_qf);
          // set boundary conditions TODO
          for (auto dd: ess_dofs_s)
            B_T[dd] = 0;
          dX_T = 0.0;
          cg_T.Mult(B_T, dX_T);

          X_T += dX_T;
          dX_T += X_T;
          

          // ********************************************
          dphi_qf.ProjectGradient(dX_0, Vs, Vt);
          ds_qf.ProjectValue(dX_0, Vs, Vt);
          dsigma_qf.ProjectValue(dX_1, Ss, St);
          divs_qf.ProjectDivergence(dX_1, Ss, St);
          dxi_qf.ProjectValue(dX_2, Vs, St);
          gradxi_qf.ProjectSpaceGradient(dX_2, Vs, St);
          dtheta_qf.ProjectValue(dX_3, Ss, St);
          divt_qf.ProjectDivergence(dX_3, Ss, St);
          if (ho){
              // add gma*div(sigma) to last component (time derivative) of dphi
              // add gma*grad(xi) to dsigma
              // add gma*div(theta) to xi
              for (int i = 0; i < nQ; ++i)
              {
                 dphi_qf[dim - 1 + i*dim] += gma * divs_qf[i];
                 for (int d=0; d < dim_s; d++)
                   dsigma_qf[d + i*dim_s] += gma * gradxi_qf[d+i*dim_s];
                 dxi_qf[i] += gma * divt_qf[i];
              }
          }

          // project dphi_T_qf, sigmaT & div(sigmaT)
          face_interp.Project(dX_0, dphi_T_qf, n_time);
          sigma_T_gf.SetFromTrueDofs(dX_T);
          qpt_interp(sigma_T_gf, dsigma_T_qf, QF_VALS);
          qpt_interp(sigma_T_gf, divs_T_qf,   QF_DIVS);
          

          sw_e.Start();
          // Pointwise nonlinear solve:: volumne part
          uintmax_t max_nonlin_it = 0;
          {
             double rho, rhobar, sbar, pbar, mbar2, sbar2;
             Vector mbar(dim-1), qbar(dim-1), nbar(dim-1);

             for (int j = 0; j < nQ; j++)
             {
                mbar2 = 0;
                sbar2 = 0;
                for (int d = 0; d < dim_s; ++d)
                {
                   // Note: vdim(dphi_qf)   = dim
                   //       vdim(dsigma_qf) = dim_s
                   //       vdim(u_qf)      = u_vdim
                   // Recall u = (rho, m, n)
                   // m-part
                   mbar[d] = sigma_u * dphi_qf[d + j*dim] + u_qf[d+1 + j*u_vdim];
                   mbar2 += pow(mbar[d], 2);
                   // n-part
                   nbar[d] = sigma_u * dsigma_qf[d + j*dim_s] + u_qf[dim + d + j*u_vdim];
                   // q-part
                   qbar[d] = -sigma_u * dtheta_qf[d + j*dim_s] 
                     + pq_qf[1 + d + j*pq_vdim];
                }
                // density
                rho = u_qf[j*u_vdim];
                rhobar = sigma_u * dphi_qf[dim - 1 + j*dim] + rho;
                // source
                sbar = sigma_u * ds_qf[j] + s_qf[j];
                sbar2 = sbar*sbar;
                // p-part
                pbar = -sigma_u * dxi_qf[j] + pq_qf[j*pq_vdim];

                rho_err_qf[j] = rho;
                // FIXME: rhoB may contain additional info. for interaction
                // potential, we don't use it now.
                //double rhoB = rhoB_qf[j];
                
                std::uintmax_t nonlin_it = 100;
                auto func = [rhobar, mbar2, nbar, sbar2, pbar, qbar](double x)
                {
                   return F_drop(x, rhobar, mbar2, nbar, sbar2, pbar, qbar, ho, ci);
                };

                auto res = boost::math::tools::brent_find_minima(
                   func, rmin, rmax, double_bits, nonlin_it);
                rho = res.first;
                max_nonlin_it = std::max(max_nonlin_it, nonlin_it);

                rho_err_qf[j] -= rho;

                // density update
                u_qf[j*u_vdim] = rho;
                const double v1r = V1(rho);
                const double v2r = V2(rho);
                const double v3r = V3(rho);
                const double de  = dE_drop(rho);
                const double d2e  = d2E_drop(rho);
                const double sigma_r_v1 = sigma_u * beta * beta / gma / gma *v1r;
                const double fac = 1.0 + sigma_r_v1 * (1+ d2e*d2e);
                // m & n & q update
                for (int d = 0; d < dim-1; d++)
                {
                   // m
                   u_qf[d+1 + j*u_vdim] = v1r/(sigma_u+v1r)*mbar(d);
                   // n
                   u_qf[d+dim + j*u_vdim] = (-sigma_r_v1*d2e*qbar(d)
                       +(1+sigma_r_v1)*nbar(d))/fac;
                   // q
                   pq_qf[1+d+j*pq_vdim] = ((1+sigma_r_v1*d2e*d2e)*qbar(d)
                       -sigma_r_v1*d2e*nbar(d))/fac;
                }
                // p & s updates
                pq_qf[j*pq_vdim] = (pbar-sigma_u*beta*beta*de*v2r)/(1
                    +sigma_u*beta*beta*v2r);
                s_qf[j] = v2r/(sigma_u+v2r)*sbar;
             }
          }
                
          sw_e.Stop();
          
          sw_f.Start();
          // Pointwise nonlinear solve:: boundary part
          for (int j = 0; j < nQs; ++j)
          {
             double rhobar = sigma_u * (-dphi_T_qf[j] + gma *divs_T_qf[j] )
               + rho_T_qf[j];
             double rho = rho_T_qf[j];
             double rho_T = rho_T1_qf[j];

             double rho_err = rho;

             std::uintmax_t nonlin_it = 100;
             auto func = [rhobar, rho_T](double x)
             {
                return Fb_drop(x, rhobar, rho_T, cb);
             };

             auto res = boost::math::tools::brent_find_minima(
                func, rmin, rmax, double_bits, nonlin_it);
             // update terminal density
             rho_T_qf[j] = res.first;
             rho_err -= res.first; // the error vector
             rho_T_err_qf[j] = abs(rho_err);
             
             // the n_T_qf part
             for (int d=0; d<dim_s;d++)
             {
                int idx = j*dim_s+d;
                double nbar = sigma_u * dsigma_T_qf[idx] + n_T_qf[idx];
                n_T_qf[idx] = nbar/(1.0+sigma_u * beta);
             }
          }
          sw_f.Stop();

          if ((iter+1)%iterPnt == 0)
          {  // compute L1 errors
             double error = rho_err_qf.L1Norm();
             double errorT = rho_T_err_qf.Integrate();
             
             MPI_Allreduce(MPI_IN_PLACE, &iter,  1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
             if (Mpi::Root())
             {
                 cout << scientific << setprecision(3) << left
                      << setw(6) << jko
                      << setw(6) << iter+1
                      << setw(6) << cg_0.GetNumIterations()
                      << setw(6) << cg_2.GetNumIterations()
                      << setw(6) << cg_T.GetNumIterations()
                      << fixed << setprecision(0)
                      << setw(6) << div_div_solver.GetAverageIterations()
                      << scientific << setprecision(3)
                      << setw(8) << max_nonlin_it
                      << setw(13) << scientific << error
                      << setw(13) << scientific << errorT
                      << setw(13) << sw_a.RealTime() // phi solve
                      << setw(13) << sw_b.RealTime() // sigma solve
                      << setw(13) << sw_c.RealTime() // newton step
                      << endl;
                 errorList.push_back(error);
                 errorTList.push_back(errorT);
             }
             if (ho)
                 // PDHG tolerance 
                 if ((error < 1e-6 || errorT < 1e-6) && iter > 20)
                   break;
             else
                 // PDHG tolerance 
                 if ((error < 5e-6 || errorT < 5e-6) && iter > 20)
                   break;
          }
          if ((iter+1)%100==0 && ho){ // visualize every 1000 iterations
            vis(false, jko);
            vis(true, jko);
          }
       }
       
       if (ho){
          vis(false, jko);
          vis(true, jko);
       } else {// JKO save every 20 steps
           if ((jko+1)%20==0){
              vis(false, jko);
              vis(true, jko);
           }
       }
       if (Mpi::Root()){
           // save PDHG converge history
           std::ofstream outputFile("ParaView/DROP_"+fileX+"_"+to_string(jko)+"/V.txt");
           // Write vector elements to the file
           for (auto value : errorList) {
               outputFile << value << endl;
           }
           // Close the file
           outputFile.close();
           
           std::ofstream outputFileT("ParaView/DROP_"+fileX+"_"+to_string(jko)+"/B.txt");
           // Write vector elements to the file
           for (auto value : errorTList) {
               outputFileT << value << endl;
           }
           // Close the file
           outputFileT.close();
       }

       B_bdr_0 =  B_bdr_T;
       B_bdr_0 *= -1;
   }

   return 0;
}
