#ifndef SURF_RHS_HPP
#define SURF_RHS_HPP

#include "mfem.hpp"
#include "kron_quad.hpp"

namespace mfem
{

class KroneckerSurfLinearForm : public Vector
{
   ParFiniteElementSpace &Vs, &Vt;
   const int dim_s;

   QuadratureSpace &Qs, &Qt;

   ParLinearForm Ls_grad, Ls_interp, Lt_grad, Lt_interp;
   KroneckerQuadratureFunction qf_s; // Dot product with the spatial gradient
   KroneckerQuadratureFunction qf_t; // Coefficient of the time derivative
   KroneckerQuadratureFunction qf_scalar; // Coefficient of the source term

   QuadratureFunction qs, qs_vec, qt;
   QuadratureFunctionCoefficient qs_coeff, qt_coeff;
   VectorQuadratureFunctionCoefficient qs_vec_coeff, qt_vec_coeff;

   Vector z1, z2, z3, z4;

   void Assemble();

public:
   KroneckerSurfLinearForm(ParFiniteElementSpace &Vs_, ParFiniteElementSpace &Vt_,
                       QuadratureSpace &Qs_, QuadratureSpace &Qt_);

   void Update(KroneckerQuadratureFunction &u_qf,
               KroneckerQuadratureFunction &s_qf);

   void Update(KroneckerQuadratureFunction &u_qf);

   using Vector::operator=;
};

} // namespace mfem

#endif
