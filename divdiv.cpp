#include "divdiv.hpp"
#include "wass_params.hpp"

namespace mfem
{

KroneckerDivDiv::KroneckerDivDiv(
   ParFiniteElementSpace &Vs, ParFiniteElementSpace &Vt,
   const Array<int> &ess_bdr_s)
:  Ds(&Vs),
   // Ms(&Vs),
   Mt(&Vt),
   Ds_coeff(gma*gma),
   nVs(Vs.GetTrueVSize()),
   nVt(Vt.GetTrueVSize())
{
   height = width = nVs*nVt;
   const int dim = Vs.GetMesh()->Dimension();

   Vs.GetEssentialTrueDofs(ess_bdr_s, ess_dofs_s);

   const bool hdiv = (Vs.FEColl()->GetMapType(dim) == FiniteElement::H_DIV);

   if (hdiv)
   {
      Ds.AddDomainIntegrator(new DivDivIntegrator(Ds_coeff));
      Ds.AddDomainIntegrator(new VectorFEMassIntegrator);
      // PA not available for surface mesh
      //if (dim >= 2) { Ds.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   }
   else
   {
      Ds.AddDomainIntegrator(new VectorDivDivIntegrator(Ds_coeff));
      Ds.AddDomainIntegrator(new VectorMassIntegrator);
   }
   Ds.Assemble();
   Ds.FormSystemMatrix(ess_dofs_s, Ds_op);

   Mt.AddDomainIntegrator(new MassIntegrator);
   Mt.Assemble();
   Mt.FormSystemMatrix(empty, Mt_op);
}

void KroneckerDivDiv::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == nVs*nVt, "");
   MFEM_ASSERT(y.Size() == nVs*nVt, "");

   kron_mult.Mult(*Mt_op, *Ds_op, x, y);
   // y *= gma*gma; // gamma scaling
   // kron_mult.AddMult(*Mt_op, *Ms_op, x, y);
}

void KroneckerDivDiv::AssembleDiagonal(Vector &diag) const
{
   // Vector diag_Ms(nVs), diag_Ds(nVs), diag_Mt(nVt);
   Vector diag_Ds(nVs), diag_Mt(nVt);

   // Ms_op->AssembleDiagonal(diag_Ms);
   Ds_op->AssembleDiagonal(diag_Ds);
   // Mt_op.As<SparseMatrix>()->GetDiag(diag_Mt);
   Mt_op->AssembleDiagonal(diag_Mt);

   diag.SetSize(nVs*nVt);

   for (int it = 0; it < nVt; ++it)
   {
      const double dMt = diag_Mt[it];
      for (int is = 0; is < nVs; ++is)
      {
         // const double dMs = diag_Ms[is];
         const double dDs = diag_Ds[is];
         // diag[is + it*nVs] = dMt*dMs + gma*gma*dMt*dDs;
         diag[is + it*nVs] = dMt*dDs;
      }
   }
}

void VectorDivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Tr, DenseMatrix &elmat)
{
   const int ndof = el.GetDof();
   const int sdim = Tr.GetSpaceDim();

   elmat.SetSize(ndof*sdim);
   dshape.SetSize(ndof, sdim); // physical gradient

   const IntegrationRule &ir = [&]()
   {
      if (IntRule) { return *IntRule; }
      int order = 2 * el.GetOrder() + 1;
      return IntRules.Get(el.GetGeomType(), order);
   }();

   elmat = 0.0;
   for (int iq = 0; iq < ir.GetNPoints(); ++iq)
   {
      const IntegrationPoint &ip = ir[iq];
      Tr.SetIntPoint(&ip);
      const double coeff_val = coeff ? coeff->Eval(Tr, ip) : 1.0;
      el.CalcPhysDShape(Tr, dshape);
      Vector dshape_vec(dshape.ReadWrite(), ndof*sdim);
      const double w = ip.weight * Tr.Weight() * coeff_val;
      AddMult_a_VVt(w, dshape_vec, elmat);
   }
}

DivDivSaddlePointSolver::DivDivSaddlePointSolver(
   ParFiniteElementSpace &rt_fes,
   Coefficient &Ds_coeff,
   const Array<int> &ess_rt_dofs)
: DivDivSolver(rt_fes.GetTrueVSize()),
  l2_fec(rt_fes.GetElementOrder(0) - 1, rt_fes.GetMesh()->Dimension()),
  l2_fes(rt_fes.GetParMesh(), &l2_fec),
  one(1.0),
  saddle_point_solver(*rt_fes.GetParMesh(), rt_fes, l2_fes, Ds_coeff, one, ess_rt_dofs, HdivSaddlePointSolver::GRAD_DIV),
  X_block(saddle_point_solver.GetOffsets()),
  B_block(saddle_point_solver.GetOffsets())
{
   B_block.GetBlock(0) = 0.0;
   X_block = 0.0;
   // NOTE: assuming homogeneous Dirichlet conditions here!

   saddle_point_solver.GetMINRES().SetRelTol(cgrtol);
   saddle_point_solver.GetMINRES().SetAbsTol(cgatol);

   saddle_point_solver.SetBC(X_block.GetBlock(1));
}

void DivDivSaddlePointSolver::Mult(const Vector &b, Vector &x) const
{
   B_block.GetBlock(1) = b;
   B_block.GetBlock(1) *= -1.0;
   B_block.SyncFromBlocks();
   X_block = 0.0;
   saddle_point_solver.Mult(B_block, X_block);
   X_block.SyncToBlocks();
   x = X_block.GetBlock(1);
}

void IterationCounter::MonitorSolution(int it, double norm, const Vector &x, bool final)
{
   if (final)
   {
      ++n;
      avg_it = avg_it + (it - avg_it)/n;
   }
}

DivDivCGSolver::DivDivCGSolver(
   ParBilinearForm &Ds,
   OperatorHandle &Ds_op,
   const Array<int> &ess_dofs,
   bool jacobi)
  : DivDivSolver(Ds.FESpace()->GetTrueVSize()),
    cg(Ds.ParFESpace()->GetComm())
{
   cg.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg.SetMaxIter(cgi);
   cg.SetRelTol(cgrtol);
   cg.SetAbsTol(cgatol);
   cg.SetOperator(*Ds_op);
   cg.SetMonitor(it_counter);

   const auto &fes = *Ds.FESpace();
   const int dim = fes.GetMesh()->Dimension();
   const bool hdiv = (fes.FEColl()->GetMapType(dim) == FiniteElement::H_DIV);

   if (jacobi || !hdiv)
   {
      prec.reset(new OperatorJacobiSmoother(Ds, ess_dofs));
   }
   else
   {
      if (dim == 1)
      {
         auto *amg = new HypreBoomerAMG(*Ds_op.As<HypreParMatrix>());
         amg->SetPrintLevel(0);
         prec.reset(amg);
      }
      else if (dim == 2)
      {
         prec.reset(new LORSolver<HypreAMS>(Ds, ess_dofs));
      }
      else // dim == 3
      {
         prec.reset(new LORSolver<HypreADS>(Ds, ess_dofs));
      }
   }
   cg.SetPreconditioner(*prec);
}

void DivDivCGSolver::Mult(const Vector &b, Vector &x) const
{
   cg.Mult(b, x);
}

KroneckerDivDivSolver::KroneckerDivDivSolver(
   ParFiniteElementSpace &Vs,
   ParFiniteElementSpace &Vt,
   const Array<int> &ess_bdr_s,
   bool jacobi)
: Solver(Vs.GetTrueVSize()*Vt.GetTrueVSize()),
  dim(Vs.GetMesh()->Dimension()),
  kron_divdiv(Vs, Vt, ess_bdr_s),
  mass_inv(*kron_divdiv.Mt_op.As<HypreParMatrix>(), HypreSmoother::Jacobi)
{
   const bool hdiv = (Vs.FEColl()->GetMapType(dim) == FiniteElement::H_DIV);

   if (hdiv && dim == 3 && !jacobi)
   {
      divdiv_solver.reset(new DivDivSaddlePointSolver(Vs, kron_divdiv.Ds_coeff, kron_divdiv.ess_dofs_s));
   }
   else
   {
      divdiv_solver.reset(new DivDivCGSolver(kron_divdiv.Ds, kron_divdiv.Ds_op, kron_divdiv.ess_dofs_s, jacobi));
   }
   // set the cg_mass 
   cg_mass.SetOperator(*kron_divdiv.Mt_op);
}

void KroneckerDivDivSolver::Mult(const Vector &x, Vector &y) const
{
   kron_divdiv.kron_mult.Mult(mass_inv, *divdiv_solver, x, y);
   //kron_divdiv.kron_mult.Mult(cg_mass, *divdiv_solver, x, y);
}

double KroneckerDivDivSolver::GetAverageIterations() const
{
   return divdiv_solver->GetAverageIterations();
}

} // namespace mfem
