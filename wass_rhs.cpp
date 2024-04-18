#include "wass_rhs.hpp"
#include "wass_params.hpp"

namespace mfem
{

KroneckerLinearForm::KroneckerLinearForm(
   ParFiniteElementSpace &Vs_, ParFiniteElementSpace &Vt_,
   QuadratureSpace &Qs_, QuadratureSpace &Qt_,
   VectorCoefficient *neumann_data)
   : Vector(Vs_.GetTrueVSize() * Vt_.GetTrueVSize()),
     Vs(Vs_),
     Vt(Vt_),
     dim_s(Vs.GetMesh()->Dimension()),
     Qs(Qs_),
     Qt(Qt_),
     Ls_grad(&Vs),
     Ls_interp(&Vs),
     Lt_grad(&Vt),
     Lt_interp(&Vt),
     qf_s(Qs, Qt, dim_s),
     qf_t(Qs, Qt, 1),
     qf_scalar(Qs, Qt, 1),
     qs(Qs),
     qs_vec(Qs, dim_s),
     qt(Qt),
     qs_coeff(qs),
     qt_coeff(qt),
     qs_vec_coeff(qs_vec),
     qt_vec_coeff(qt),
     z1(Vs.GetTrueVSize()*Qt.GetSize()),
     z2(Vs.GetTrueVSize()*Qt.GetSize()),
     z3(Vt.GetTrueVSize()),
     z4(Vs.GetTrueVSize())
{
   auto set_int_rule = [&](ParLinearForm &lf, const IntegrationRule &ir)
   {
      for (auto *i : *lf.GetDLFI()) { i->SetIntRule(&ir); }
   };

   Ls_grad.AddDomainIntegrator(new DomainLFGradIntegrator(qs_vec_coeff));
   if (neumann_data)
   {
      Ls_grad.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*neumann_data));
   }
   Ls_interp.AddDomainIntegrator(new DomainLFIntegrator(qs_coeff));

   Lt_grad.AddDomainIntegrator(new DomainLFGradIntegrator(qt_vec_coeff));
   Lt_interp.AddDomainIntegrator(new DomainLFIntegrator(qt_coeff));

   set_int_rule(Ls_grad, Qs.GetIntRule(0));
   set_int_rule(Ls_interp, Qs.GetIntRule(0));
   set_int_rule(Lt_grad, Qt.GetIntRule(0));
   set_int_rule(Lt_interp, Qt.GetIntRule(0));
}

void KroneckerLinearForm::Assemble()
{
   const int nQs = Qs.GetSize();
   const int nQt = Qt.GetSize();

   const int nVs = Vs.GetTrueVSize();
   const int nVt = Vt.GetTrueVSize();

   HostWrite();

   // Compute space integrals for each time quadrature point
   for (int it = 0; it < nQt; ++it)
   {
      qs_vec.MakeRef(qf_s, it*nQs*dim_s, nQs*dim_s);
      qs.MakeRef(qf_t, it*nQs, nQs);

      Vector z1_slice(z1, it*nVs, nVs);
      Vector z2_slice(z2, it*nVs, nVs);

      Ls_grad.Assemble();
      Ls_grad.ParallelAssemble(z1_slice);
      Ls_interp.Assemble();
      Ls_interp.ParallelAssemble(z2_slice);

      // Add source term
      qs.MakeRef(qf_scalar, it*nQs, nQs);
      Ls_interp.Assemble();
      Ls_interp.ParallelAssemble(z4);
      z1_slice += z4;
   }

   for (int is = 0; is < nVs; ++is)
   {
      for (int it = 0; it < nQt; ++it)
      {
         qt[it] = z1[is + it*nVs];
      }
      Lt_interp.Assemble();
      Lt_interp.ParallelAssemble(z3);

      z3.HostReadWrite();

      for (int it = 0; it < nVt; ++it)
      {
         (*this)[is + it*nVs] = z3[it];
      }

      for (int it = 0; it < nQt; ++it)
      {
         qt[it] = z2[is + it*nVs];
      }
      Lt_grad.Assemble();
      Lt_grad.ParallelAssemble(z3);

      z3.HostReadWrite();

      for (int it = 0; it < nVt; ++it)
      {
         (*this)[is + it*nVs] += z3[it];
      }
   }
}

void KroneckerLinearForm::Update(
   KroneckerQuadratureFunction &u_qf, KroneckerQuadratureFunction &s_qf)
{
   const int nq = qf_t.Size();
   const int u_vdim = u_qf.GetVDim();

   for (int i = 0; i < nq; ++i)
   {
      qf_scalar[i] = -s_qf[i];
      for (int d = 0; d < dim_s; ++d)
      {
         // m components
         qf_s[d + i*dim_s] = -u_qf[d + 1 + i*u_vdim];
      }
      // rho component
      qf_t[i] = -u_qf[i*u_vdim];
   }

   Assemble();
}

void KroneckerLinearForm::Update(KroneckerQuadratureFunction &u_qf)
{
   const int nq = qf_t.Size();
   const int u_vdim = u_qf.GetVDim();

   qf_scalar = 0.0;

   for (int i = 0; i < nq; ++i)
   {
      for (int d = 0; d < dim_s; ++d)
      {
         // m components
         qf_s[d + i*dim_s] = -u_qf[d + 1 + i*u_vdim];
      }
      // rho component
      qf_t[i] = -u_qf[i*u_vdim];
   }

   Assemble();
}

KroneckerFisherLinearForm::KroneckerFisherLinearForm(
   ParFiniteElementSpace &Ss_, ParFiniteElementSpace &St_,
   QuadratureSpace &Qs_, QuadratureSpace &Qt_)
   : Vector(Ss_.GetTrueVSize() * St_.GetTrueVSize()),
     Ss(Ss_),
     St(St_),
     dim_s(Ss.GetMesh()->Dimension()),
     Qs(Qs_),
     Qt(Qt_),
     Ls_div(&Ss),
     Ls_interp(&Ss),
     Lt_interp(&St),
     qf_div(Qs, Qt, 1),
     qf_vec(Qs, Qt, dim_s),
     qs(Qs),
     qs_vec(Qs, dim_s),
     qt(Qt),
     qs_coeff(qs),
     qt_coeff(qt),
     qs_vec_coeff(qs_vec),
     z1(Ss.GetTrueVSize()*Qt.GetSize()),
     z2(Ss.GetTrueVSize()),
     z3(St.GetTrueVSize())
{
   auto set_int_rule = [&](ParLinearForm &lf, const IntegrationRule &ir)
   {
      for (auto *i : *lf.GetDLFI()) { i->SetIntRule(&ir); }
   };

   const bool hdiv = Ss.FEColl()->GetMapType(dim_s) == FiniteElement::H_DIV;

   if (hdiv)
   {
      Ls_div.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(qs_coeff));
      Ls_interp.AddDomainIntegrator(new VectorFEDomainLFIntegrator(qs_vec_coeff));
   }
   else
   {
      Ls_div.AddDomainIntegrator(new DomainLFDivIntegrator(qs_coeff));
      Ls_interp.AddDomainIntegrator(new VectorDomainLFIntegrator(qs_vec_coeff));
   }

   Lt_interp.AddDomainIntegrator(new DomainLFIntegrator(qt_coeff));

   set_int_rule(Ls_div, Qs.GetIntRule(0));
   set_int_rule(Ls_interp, Qs.GetIntRule(0));
   set_int_rule(Lt_interp, Qt.GetIntRule(0));
}

void KroneckerFisherLinearForm::Assemble()
{
   const int nQs = Qs.GetSize();
   const int nQt = Qt.GetSize();

   const int nVs = Ss.GetTrueVSize();
   const int nVt = St.GetTrueVSize();

   HostWrite();

   // Compute space integrals for each time quadrature point
   for (int it = 0; it < nQt; ++it)
   {
      qs.MakeRef(qf_div, it*nQs, nQs);
      Vector z1_slice(z1, it*nVs, nVs);
      Ls_div.Assemble();
      Ls_div.ParallelAssemble(z1_slice);

      qs_vec.MakeRef(qf_vec, it*nQs*dim_s, nQs*dim_s);
      Ls_interp.Assemble();
      Ls_interp.ParallelAssemble(z2);

      z1_slice += z2;
   }

   for (int is = 0; is < nVs; ++is)
   {
      for (int it = 0; it < nQt; ++it)
      {
         qt[it] = z1[is + it*nVs];
      }
      Lt_interp.Assemble();
      Lt_interp.ParallelAssemble(z3);

      z3.HostReadWrite();

      for (int it = 0; it < nVt; ++it)
      {
         (*this)[is + it*nVs] = z3[it];
      }
   }
}

void KroneckerFisherLinearForm::Update(
   KroneckerQuadratureFunction &u_qf, KroneckerQuadratureFunction &dphi_qf)
{
   const int nq = qf_div.Size();
   const int dim = dim_s + 1;
   const int u_vdim = u_qf.GetVDim();

   for (int i = 0; i < nq; ++i)
   {
      for (int d = 0; d < dim_s; ++d)
      {
         // n components
         qf_vec[d + i*dim_s] = -u_qf[dim + d + i*u_vdim];
      }
      // rho and dphi/dt components
      qf_div[i] = -gma*u_qf[i*u_vdim] - gma*dphi_qf[dim - 1 + i*dim]/sigma_phi;
   }

   Assemble();
}

void KroneckerFisherLinearForm::Update(
   KroneckerQuadratureFunction &pq_qf, KroneckerQuadratureFunction &dxi_qf,
   bool flag)
{
   const int nq = qf_div.Size();
   const int dim = dim_s + 1;
   const int pq_vdim = pq_qf.GetVDim();

   for (int i = 0; i < nq; ++i)
   {
      for (int d = 0; d < dim_s; ++d)
      {
         // q components
         qf_vec[d + i*dim_s] = pq_qf[1 + d + i*pq_vdim];
      }
      // p and dxi components
      qf_div[i] = gma*pq_qf[i*pq_vdim] - gma*dxi_qf[i]/sigma_phi;
   }

   Assemble();
}

void DomainLFDivIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   const int ndof = el.GetDof();
   const int sdim = Tr.GetSpaceDim();

   elvect.SetSize(ndof*sdim);
   dshape.SetSize(ndof, sdim);// the physical gradient

   // divshape.SetSize(nd*vdim);   // the spatial divergence

   const IntegrationRule &ir = [&]()
   {
      if (IntRule) { return *IntRule; }
      int order = 2 * el.GetOrder() + 1;
      return IntRules.Get(el.GetGeomType(), order);
   }();

   elvect = 0.0;
   for (int iq = 0; iq < ir.GetNPoints(); ++iq)
   {
      const IntegrationPoint &ip = ir[iq];
      Tr.SetIntPoint (&ip);

      el.CalcPhysDShape(Tr, dshape);

      const double w = ip.weight * Tr.Weight(); // integration weight
      const double q = coeff.Eval(Tr, ip);

      // Spatial divergence rhs
      for (int vd = 0; vd < sdim; ++vd)
      {
         for (int i = 0; i < ndof; ++i)
         {
            const int idx = i + vd*ndof;
            elvect[idx] += w * q * dshape(i, vd);
         }
      }
   }
}

// Xi RHS
KroneckerXiLinearForm::KroneckerXiLinearForm(
   ParFiniteElementSpace &Vs_, ParFiniteElementSpace &Vt_,
   QuadratureSpace &Qs_, QuadratureSpace &Qt_)
   : Vector(Vs_.GetTrueVSize() * Vt_.GetTrueVSize()),
     Vs(Vs_),
     Vt(Vt_),
     dim_s(Vs.GetMesh()->Dimension()),
     Qs(Qs_),
     Qt(Qt_),
     Ls_grad(&Vs),
     Ls_interp(&Vs),
     Lt_interp(&Vt),
     qf_s(Qs, Qt, dim_s),
     qf_scalar(Qs, Qt, 1),
     qs(Qs),
     qs_vec(Qs, dim_s),
     qt(Qt),
     qs_coeff(qs),
     qt_coeff(qt),
     qs_vec_coeff(qs_vec),
     z1(Vs.GetTrueVSize()*Qt.GetSize()),
     z2(Vs.GetTrueVSize()),
     z3(Vt.GetTrueVSize())
{
   auto set_int_rule = [&](ParLinearForm &lf, const IntegrationRule &ir)
   {
      for (auto *i : *lf.GetDLFI()) { i->SetIntRule(&ir); }
   };

   Ls_grad.AddDomainIntegrator(new DomainLFGradIntegrator(qs_vec_coeff));
   Ls_interp.AddDomainIntegrator(new DomainLFIntegrator(qs_coeff));

   Lt_interp.AddDomainIntegrator(new DomainLFIntegrator(qt_coeff));

   set_int_rule(Ls_grad, Qs.GetIntRule(0));
   set_int_rule(Ls_interp, Qs.GetIntRule(0));
   set_int_rule(Lt_interp, Qt.GetIntRule(0));
}

void KroneckerXiLinearForm::Assemble()
{
   const int nQs = Qs.GetSize();
   const int nQt = Qt.GetSize();

   const int nVs = Vs.GetTrueVSize();
   const int nVt = Vt.GetTrueVSize();

   HostWrite();

   // Compute space integrals for each time quadrature point
   for (int it = 0; it < nQt; ++it)
   {
      qs_vec.MakeRef(qf_s, it*nQs*dim_s, nQs*dim_s);
      Vector z1_slice(z1, it*nVs, nVs);
      Ls_grad.Assemble();
      Ls_grad.ParallelAssemble(z1_slice);

      // Add source term
      qs.MakeRef(qf_scalar, it*nQs, nQs);
      Ls_interp.Assemble();
      Ls_interp.ParallelAssemble(z2);
      z1_slice += z2;
   }

   for (int is = 0; is < nVs; ++is)
   {
      for (int it = 0; it < nQt; ++it)
      {
         qt[it] = z1[is + it*nVs];
      }
      Lt_interp.Assemble();
      Lt_interp.ParallelAssemble(z3);

      z3.HostReadWrite();

      for (int it = 0; it < nVt; ++it)
      {
         (*this)[is + it*nVs] = z3[it];
      }
   }
}

void KroneckerXiLinearForm::Update(
   KroneckerQuadratureFunction &u_qf, 
   KroneckerQuadratureFunction &pq_qf, 
   KroneckerQuadratureFunction &dsigma_qf)
{
   const int nq = qf_scalar.Size();
   const int u_vdim = u_qf.GetVDim();
   const int pq_vdim = pq_qf.GetVDim();
   const int ds_vdim = dsigma_qf.GetVDim();

   for (int i = 0; i < nq; ++i)
   {
      qf_scalar[i] = pq_qf[i*pq_vdim];
      for (int d = 0; d < dim_s; ++d)
      {
         // n components
         qf_s[d + i*dim_s] = -gma*(u_qf[dim_s + d + 1 + i*u_vdim]
             + dsigma_qf[d+i*ds_vdim]/sigma_phi);
      }
   }

   Assemble();
}


// divT-rhs
DivTLinearForm::DivTLinearForm(ParFiniteElementSpace &Ss_, 
    QuadratureSpace &Qs_)
   : Vector(Ss_.GetTrueVSize()),
     Ss(Ss_),
     dim_s(Ss.GetMesh()->Dimension()),
     Qs(Qs_),
     Ls_div(&Ss),
     Ls_interp(&Ss),
     qs(Qs),
     qs_vec(Qs, dim_s),
     qs_coeff(qs),
     qs_vec_coeff(qs_vec),
     z1(Ss.GetTrueVSize()),
     z2(Ss.GetTrueVSize())
{
   auto set_int_rule = [&](ParLinearForm &lf, const IntegrationRule &ir)
   {
      for (auto *i : *lf.GetDLFI()) { i->SetIntRule(&ir); }
   };

   Ls_div.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(qs_coeff));
   Ls_interp.AddDomainIntegrator(new VectorFEDomainLFIntegrator(qs_vec_coeff));

   set_int_rule(Ls_div, Qs.GetIntRule(0));
   set_int_rule(Ls_interp, Qs.GetIntRule(0));
}

void DivTLinearForm::Assemble()
{
   const int nQs = Qs.GetSize();
   const int nVs = Ss.GetTrueVSize();

   HostWrite();

   // qs
   Ls_div.Assemble();
   Ls_div.ParallelAssemble(z1);

   // qs_vec
   Ls_interp.Assemble();
   Ls_interp.ParallelAssemble(z2);

   for (int it = 0; it < nVs; ++it)
   {
      (*this)[it] = z1[it]+z2[it];
   }
}

void DivTLinearForm::Update(QuadratureFunction &rho_T_qf, 
    QuadratureFunction &n_T_qf,
    QuadratureFunction &dphi_T_qf)
{
   const int nq = qs.Size();
   const int dim = dim_s + 1;

   for (int i = 0; i < nq; ++i)
   {
      for (int d = 0; d < dim_s; ++d)
      {
         // n components
         qs_vec[d + i*dim_s] = -n_T_qf[d + i*dim_s];
      }
      // rho and dphi/dt components
      qs[i] = -gma*(rho_T_qf[i] - dphi_T_qf[i]/sigma_phi);
   }

   Assemble();
}

} // namespace mfem
