#ifndef KRON_LAPLACE_HPP
#define KRON_LAPLACE_HPP

#include "mfem.hpp"
#include "kron_mult.hpp"

namespace mfem
{

class KroneckerLaplacian : public Operator
{
   ParBilinearForm Ls, Ms;
   BilinearForm Lt, Mt;
   const int nVs, nVt;
   const double mass_coeff;
   const int bdr_offset;

   OperatorHandle Ls_op, Ms_op, Lt_op, Mt_op;

   Array<int> empty;

   KronMult kron_mult;
   mutable Vector z;
public:
   KroneckerLaplacian(ParFiniteElementSpace &Vs,
                      FiniteElementSpace &Vt,
                      const double mass_coeff_,
                      const int bdr_offset_);

   void Mult(const Vector &x, Vector &y) const override;

   void AssembleDiagonal(Vector &diag) const override;
};


// TODO: how to impose essential boundary conditions?! ?!
class KroneckerSpaceLaplacian : public Operator
{
   ParBilinearForm Ls, Ms;
   BilinearForm Mt;
   const int nVs, nVt;

   OperatorHandle Ls_op, Ms_op, Mt_op;

   Array<int> ess_dofs_v, empty;

   KronMult kron_mult;
   mutable Vector z;
public:
   KroneckerSpaceLaplacian(ParFiniteElementSpace &Vs,
                      FiniteElementSpace &Vt,
                      const Array<int> &ess_bdr_v);

   void Mult(const Vector &x, Vector &y) const override;

   void AssembleDiagonal(Vector &diag) const override;
};

}

#endif
