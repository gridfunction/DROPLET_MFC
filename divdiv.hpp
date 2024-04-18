#ifndef DIVDIV_HPP
#define DIVDIV_HPP

#include "mfem.hpp"
#include "kron_mult.hpp"
#include "../hdiv_solver/hdiv_linear_solver.hpp"

namespace mfem
{

class KroneckerDivDiv : public Operator
{
   friend class KroneckerDivDivSolver;

   ParBilinearForm Ds;
   ParBilinearForm Mt;
   mutable ConstantCoefficient Ds_coeff;
   const int nVs, nVt;

   OperatorHandle Ds_op, Mt_op;

   Array<int> ess_dofs_s, empty;

   KronMult kron_mult;
public:
   KroneckerDivDiv(
      ParFiniteElementSpace &Vs,
      ParFiniteElementSpace &Vt,
      const Array<int> &ess_bdr_s);

   void Mult(const Vector &x, Vector &y) const override;

   void AssembleDiagonal(Vector &diag) const override;
};

class VectorDivDivIntegrator : public BilinearFormIntegrator
{
   Coefficient *coeff;
   DenseMatrix dshape;
public:
   VectorDivDivIntegrator() : coeff(nullptr) { }
   VectorDivDivIntegrator(Coefficient &coeff_) : coeff(&coeff_) { }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);
};

class DivDivSolver : public Solver
{
public:
   DivDivSolver(int size) : Solver(size) { }
   virtual void ResetAverageIterations() const = 0;
   virtual double GetAverageIterations() const = 0;
};

class DivDivSaddlePointSolver : public DivDivSolver
{
   L2_FECollection l2_fec;
   ParFiniteElementSpace l2_fes;
   ConstantCoefficient one;
   HdivSaddlePointSolver saddle_point_solver;

   mutable BlockVector X_block, B_block;

public:
   DivDivSaddlePointSolver(ParFiniteElementSpace &rt_fes, Coefficient &Ds_coeff, const Array<int> &ess_rt_dofs);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override { };
   void ResetAverageIterations() const override { saddle_point_solver.ResetAverageIterations(); }
   double GetAverageIterations() const override { return saddle_point_solver.GetAverageIterations(); }
};

class IterationCounter : public IterativeSolverMonitor
{
   int n = 0;
   double avg_it = 0;
public:
   void MonitorSolution(int it, double norm, const Vector &x, bool final) override;
   void ResetAverageIterations() { n = 0; avg_it = 0; }
   double GetAverageIterations() const { return avg_it; }
};

class DivDivCGSolver : public DivDivSolver
{
   CGSolver cg;
   mutable IterationCounter it_counter;
   std::unique_ptr<Solver> prec;
public:
   DivDivCGSolver(ParBilinearForm &Ds, OperatorHandle &Ds_op, const Array<int> &ess_dofs, bool jacobi);
   void Mult(const Vector &b, Vector &x) const override;
   void SetOperator(const Operator &op) override { };
   void ResetAverageIterations() const override { it_counter.ResetAverageIterations(); }
   double GetAverageIterations() const override { return it_counter.GetAverageIterations(); }
};

class KroneckerDivDivSolver : public Solver
{
   const int dim;
   KroneckerDivDiv kron_divdiv;

   HypreSmoother mass_inv;
   std::unique_ptr<DivDivSolver> divdiv_solver;
   CGSolver cg_mass;
public:
   KroneckerDivDivSolver(
      ParFiniteElementSpace &Vs,
      ParFiniteElementSpace &Vt,
      const Array<int> &ess_bdr_s,
      bool jacobi = false);

   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &op) override { }
   double GetAverageIterations() const;
};

}

#endif
