#ifndef WASS_RHS_HPP
#define WASS_RHS_HPP

#include "mfem.hpp"
#include "kron_quad.hpp"

namespace mfem
{

class KroneckerLinearForm : public Vector
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
   KroneckerLinearForm(ParFiniteElementSpace &Vs_, ParFiniteElementSpace &Vt_,
                       QuadratureSpace &Qs_, QuadratureSpace &Qt_,
                       VectorCoefficient *neumann_data = nullptr);

   void Update(KroneckerQuadratureFunction &u_qf,
               KroneckerQuadratureFunction &s_qf);

   void Update(KroneckerQuadratureFunction &u_qf);

   using Vector::operator=;
};

class KroneckerFisherLinearForm : public Vector
{
   ParFiniteElementSpace &Ss, &St;
   const int dim_s;

   QuadratureSpace &Qs, &Qt;

   ParLinearForm Ls_div, Ls_interp, Lt_interp;
   KroneckerQuadratureFunction qf_div; // Coefficient of the divergence term
   KroneckerQuadratureFunction qf_vec; // Coefficient of the vector term

   QuadratureFunction qs, qs_vec, qt;
   QuadratureFunctionCoefficient qs_coeff, qt_coeff;
   VectorQuadratureFunctionCoefficient qs_vec_coeff;

   Vector z1, z2, z3;

   void Assemble();

public:
   KroneckerFisherLinearForm(
      ParFiniteElementSpace &Ss_, ParFiniteElementSpace &St_,
      QuadratureSpace &Qs_, QuadratureSpace &Qt_);

   // sigma RHS
   void Update(KroneckerQuadratureFunction &u_qf,
               KroneckerQuadratureFunction &dphi_qf);
   // theta RHS
   void Update(KroneckerQuadratureFunction &pq_qf,
               KroneckerQuadratureFunction &dxi_qf, 
               bool flag);
   
   using Vector::operator=;
};

/// Class for domain integrator L(v) := (f, div v) where v = (v_1, ..., v_d),
/// and each v_i is in a scalar finite element space.
class DomainLFDivIntegrator : public LinearFormIntegrator
{
private:
   Coefficient &coeff;
   DenseMatrix dshape, dshape_phys;

public:
   DomainLFDivIntegrator(Coefficient &coeff_) : coeff(coeff_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


// Xi RHS
class KroneckerXiLinearForm : public Vector
{
   ParFiniteElementSpace &Vs, &Vt;
   const int dim_s;

   QuadratureSpace &Qs, &Qt;

   ParLinearForm Ls_grad, Ls_interp, Lt_interp;
   KroneckerQuadratureFunction qf_s; // Dot product with the spatial gradient
   KroneckerQuadratureFunction qf_scalar; // Coefficient of the source term

   QuadratureFunction qs, qs_vec, qt;
   QuadratureFunctionCoefficient qs_coeff, qt_coeff;
   VectorQuadratureFunctionCoefficient qs_vec_coeff;

   Vector z1, z2, z3;

   void Assemble();

public:
   KroneckerXiLinearForm(
      ParFiniteElementSpace &Vs_, ParFiniteElementSpace &Vt_,
      QuadratureSpace &Qs_, QuadratureSpace &Qt_);

   void Update(KroneckerQuadratureFunction &u_qf,
               KroneckerQuadratureFunction &pq_qf,
               KroneckerQuadratureFunction &dsigma_qf);

   using Vector::operator=;
};

// divT-rhs
class DivTLinearForm : public Vector
{
   ParFiniteElementSpace &Ss;
   const int dim_s;

   QuadratureSpace &Qs;

   ParLinearForm Ls_div, Ls_interp;

   QuadratureFunction qs, qs_vec;
   QuadratureFunctionCoefficient qs_coeff;
   VectorQuadratureFunctionCoefficient qs_vec_coeff;

   Vector z1, z2, z3;

   void Assemble();

public:
   DivTLinearForm(ParFiniteElementSpace &Ss_, QuadratureSpace &Qs_);

   // sigma RHS
   void Update(QuadratureFunction &rho_T_qf,
               QuadratureFunction &n_T_qf,
               QuadratureFunction &dphi_T_qf);
   
   using Vector::operator=;
};


} // namespace mfem

#endif
