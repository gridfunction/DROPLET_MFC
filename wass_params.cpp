#include "wass_params.hpp"

using namespace mfem;

double rho_ex(const Vector &x, double t)
{
   const int dim_s = x.Size(); // spatial dimension
   const int dim = dim_s + 1; // space-time dimension
   if (typeI==0)
   {
      double fac = 1.0;
      if (c2>0) { fac = c2; }
      if (dim == 2)
      {
         return 1.0 + fac*0.5*cos(2*M_PI*x[0])*exp(-(4*M_PI*M_PI+c2)*beta*t);
      }
      else
      {
         return 1.0 + fac*0.5*cos(2*M_PI*x[0])*cos(2*M_PI*x[1])
            *exp(-(8*M_PI*M_PI+c2)*beta*t);
      }
   }
   else if (typeI==1)
   {
       if (dim==2)
       {
           return exp(-100*(x[0]-0.5)*(x[0]-0.5));
       }
       else
       {
           return exp(-100*((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)));
       }
   }
   else { MFEM_ABORT(""); }
   return 0.0;
}

double rho_sbp_ex(const Vector &x, double t)
{
   if (test_case/10 == 1)
   {
      int dim = x.Size();
      double xc = 0.25 + 0.5*t, yc = 0.25 + 0.5*t, zc = 0.25 + 0.5*t;
      double r2 = pow(x(0)-xc,2)+pow(x(1)-yc,2);
      if (dim==3)
          r2 += pow(x(2)-zc,2);
      return exp(-100*r2);
      //return pow(x(0)-0.5, 2) + pow(x(1)-0.5,2) + 0.1;
      //return 1.0 + 0.2*cos(4.0*M_PI*x(0));
   }
   else if (test_case/10 == 2)
   {
      int dim = x.Size();
      double xc = 0.25 + 2.0*t, yc = 0.5, zc = 0.5;
      double r2 = pow(x(0)-xc,2)+pow(x(1)-yc,2);
      if (dim==3)
          r2 += pow(x(2)-zc,2);
      return exp(-100*r2);
   }
   else if (test_case == 3)
   {
      double xc = (t >0.5) ? 0.88: -0.81;
      if (t>0.5)
          return (x[0] > xc) ? 1.0: 1e-4;
      else
          return (x[0] < xc) ? 1.0: 1e-4;
   }
   else if (test_case == 4) //3D cube
   {
      //double xc = (t >0.5) ? 0.75: 0.25;
      //if (t>0.5)
      //    return (x[0] > xc) ? 1.0: 1e-4;
      //else
      //    return (x[0] < xc) ? 1.0: 1e-4;
      double xc = 0.25 + 2.0*t, yc = 0.5;
      double r2 = pow(x(0)-xc,2)+pow(x(1)-yc,2)+pow(x(2)-0.5, 2);
      double val = exp(-100*r2);
      return val;
   }else{
     return 1.0 + 0.5*cos(4.0*M_PI*x(0));
   }
}

double rho_ccl_ex(const Vector &x, double t)
{
   return 1.0 + 0.5*sin(2.0*M_PI*(x(0)-0.2*t));
}

double rho_sbp_surf_ex(const Vector &x, double t)
{
  double theta = (t < 0.5) ? 0.0 : M_PI;
  double xc = cos(theta), yc = sin(theta), zc = 0.0;
  double r2 = pow(x(0)-xc,2)+pow(x(1)-yc,2)+pow(x(2)-zc,2);
  if (test_case==1)
    return exp(-20*r2);
  else
    return exp(-200*pow(x(1)+0.5-t,2)); 
}

double rho_ccl_pde_ex(const Vector &x, double t)
{
   const int dim_s = x.Size(); // spatial dimension
   if (typeI == 0){
       if (dim_s == 1)
       {
          return 1.0 + 0.5*cos(2*M_PI*x[0])*exp(-(4*M_PI*M_PI)*beta*t);
       }
       else
       {
          return 1.0 + 0.5*sin(2*M_PI*(x[0]+x[1]))
             *exp(-(8*M_PI*M_PI)*beta*t);
       }
   }else if (typeI == 1){
       // KPP
       double val = 0.25* M_PI;
       double rad2 = (x[0]-2.0)*(x[0]-2.0) + (x[1]-2.5)*(x[1]-2.5);
       if (rad2 < 1.0)
           val += 3.25*M_PI;
       return val;
   }else if (typeI ==2){
       //Buckley-Leverett
       double val = 1e-4;
       if ((x[0]*x[0] + x[1] * x[1]) < 0.5)
           val += 1.0;
       return val;
   }else 
       return 1.0;
}

double rho_obs(const Vector &x, double t)
{
    double val1 = 0.05*0.05-(x(0)-0.6)*(x(0)-0.6) - (x(1)-0.6)*(x(1)-0.6);
    if (val1 <0) val1 = 0.0;
    else
        val1 = 1.0;
    return val1;
}

double rho_drop_ex(const Vector &x, double t)
{
  if (test_case==12)
    return 1.0 + 0.2*cos(4.0*M_PI*x(0));
  else if (test_case==22 or test_case==23 or test_case==25){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double val = 0.5*H0*(w0*w0-(x(0)-0.5)*(x(0)-0.5)/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==24){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double val1 = 0.25*H0*(w0*w0-(x(0)-0.7)*(x(0)-0.7)/0.01/0.01);
    double val2 = 0.25*H0*(w0*w0-(x(0)-0.3)*(x(0)-0.3)/0.01/0.01);
    if (val1 <0) val1 = 0.0;
    if (val2<0) val2 = 0.0;
    return eps + val1 + val2;
  }
  else if (test_case==32){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.3)*(x(1)-0.3);
    //return 10.0/3.0*exp(-300*rad2)+eps;
    double val = 0.5*H0*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==33 or test_case==35){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.5)*(x(0)-0.5) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H0*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==34){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2A = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.7)*(x(1)-0.7);
    double rad2B = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.7)*(x(1)-0.7);
    double rad2C = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.3)*(x(1)-0.3);
    double rad2D = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.3)*(x(1)-0.3);
    double val1 = 0.125*H0*(w0*w0-rad2A/0.01/0.01);
    double val2 = 0.125*H0*(w0*w0-rad2B/0.01/0.01);
    double val3 = 0.125*H0*(w0*w0-rad2C/0.01/0.01);
    double val4 = 0.125*H0*(w0*w0-rad2D/0.01/0.01);
    if (val1 <0) val1 = 0.0;
    if (val2<0) val2 = 0.0;
    if (val3<0) val3 = 0.0;
    if (val4<0) val4 = 0.0;
    return eps + val1 + val2 + val3 + val4;
  }
  else if (test_case==36){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.25)*(x(0)-0.25) + (x(1)-0.25)*(x(1)-0.25);
    double val = 0.5*H0*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==37){ // sharpening
    double fac = 2.0;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.5)*(x(0)-0.5) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H1*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==39) // JKO
    return 1.0 + 0.2*cos(2.0*M_PI*x(0))*cos(2.0*M_PI*x(1));
  else
    return 1.0 + 0.2*cos(2.0*M_PI*x(0))*cos(2*M_PI*x(1));
}

double rho_drop_target(const Vector &x)
{
  if (test_case==12)
    return 1.0 + 0.2*cos(4.0*M_PI*x(0));
  else if (test_case==22){ // transport
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double val = 0.5*H0*(w0*w0-(x(0)-0.7)*(x(0)-0.7)/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==23){ // flatten
    double fac = 1.5;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double val = 0.5*H1*(w0*w0-(x(0)-0.5)*(x(0)-0.5)/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==24){ // merge
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double val = 0.5*H0*(w0*w0-(x(0)-0.5)*(x(0)-0.5)/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==25){ // splitting
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double val1 = 0.25*H0*(w0*w0-(x(0)-0.7)*(x(0)-0.7)/0.01/0.01);
    double val2 = 0.25*H0*(w0*w0-(x(0)-0.3)*(x(0)-0.3)/0.01/0.01);
    if (val1 <0) val1 = 0.0;
    if (val2<0) val2 = 0.0;
    return eps + val1 + val2;
  }
  else if (test_case==32){ // transport
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.7)*(x(1)-0.7);
    //return 10.0/3.0*exp(-300*rad2)+eps;
    double val = 0.5*H0*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case == 36){ // transport
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.75)*(x(0)-0.75) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H0*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==33){ // flattern
    double fac = 2.0;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.5)*(x(0)-0.5) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H1*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==34 || test_case==37){ // merge & sharpening
    double fac = 1.0;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = (x(0)-0.5)*(x(0)-0.5) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H1*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case==35){
    double H0 = 0.05;
    double w0 = 1.0/sqrt(3)/H0;
    double rad2A = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.7)*(x(1)-0.7);
    double rad2B = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.7)*(x(1)-0.7);
    double rad2C = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.3)*(x(1)-0.3);
    double rad2D = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.3)*(x(1)-0.3);
    double val1 = 0.125*H0*(w0*w0-rad2A/0.01/0.01);
    double val2 = 0.125*H0*(w0*w0-rad2B/0.01/0.01);
    double val3 = 0.125*H0*(w0*w0-rad2C/0.01/0.01);
    double val4 = 0.125*H0*(w0*w0-rad2D/0.01/0.01);
    if (val1 <0) val1 = 0.0;
    if (val2<0) val2 = 0.0;
    if (val3<0) val3 = 0.0;
    if (val4<0) val4 = 0.0;
    return eps + val1 + val2 + val3 + val4;
  }
  else
    return 1.0 + 0.2*cos(2.0*M_PI*x(0))*cos(2*M_PI*x(1));
}

// initial/terminal density
double rho_0(const Vector &x)
{
   return rho_ex(x, 0.0);
}

// interaction potential
double E(double rho)
{
   if (abs(typeE-1) < 1e-4) return rho*(log(rho) - 1);// linear diffusion
   else if (typeE > 1) return pow(rho, typeE)/(typeE-1); // PME
   else { MFEM_ABORT(""); }
   return 0.0;
}

double dE(double rho)
{
   if (abs(typeE-1) < 1e-4) return log(rho);// linear diffusion
   else if (typeE > 1) return typeE*pow(rho, typeE-1)/(typeE-1); // PME
   else { MFEM_ABORT(""); }
   return 0.0;
}

double d2E(double rho)
{
   if (abs(typeE-1) < 1e-4) return 1.0/rho;// linear diffusion
   else if (typeE > 1) return typeE*pow(rho, typeE-2); // PME
   else { MFEM_ABORT(""); }
   return 0.0;
}

double E_drop(double rho)
{
  double P = 0.5;
  double val = eps/rho;
  return pow(val,3)/3 - pow(val, 2)/2 - rho * P;
}

double dE_drop(double rho)
{
  double P = 0.5;
  double val = eps/rho;
  return (-pow(val,3) + pow(val, 2))/rho - P;
}

double d2E_drop(double rho)
{
    double P = 0.5;
    double val = eps/rho;
    return (4*pow(val,3) - 3* pow(val, 2))/rho/rho;
}

double V1(double rho) { 
  if (typeV1==3)
    return pow(rho,3); 
  else
    return rho;
}

// reaction mobility
double V2(double rho)
{
  if (typeV2==0){// (rho-1)/log(rho)
    // desingularize: (1) cut off below 1e-14
    return c2*rho;
    //if (rho < 1e-14)
    //    rho = 1e-14;
    //// desingularize: (2) avoid rho=1
    //if (log(rho) == 0.0) // lim (rho-1)/log(rho) = 1
    //    return c2;
    //return c2*(rho-1.0)/log(rho);
  }else if (typeV2==1){// KPP
    // desingularize: (1) cut off below 1e-14
    if (rho < 1e-14)
        rho = 1e-14;
    // desingularize: (2) avoid rho=1
    if (log(rho) == 0.0) // lim (rho-1)/log(rho) = 1
        return c2;
    return c2*rho*(rho-1.0)/log(rho);
  }else if (typeV2==3){
   return c2/(rho+0.1);
  }
   else { MFEM_ABORT(""); }
   return 0.0;
}
// V3 = 1/V1(rho)/E''(rho)**2
double V3(double rho)
{
    return gma*gma/(beta*beta)/V1(rho)/pow(d2E(rho),2);
}

// flux = rho**2/2 (burgers)
double flux(double rho, int k, int typeF)
{
    if (typeF==0) // burgers
        return 0.5*rho*rho;
    else if (typeF==1) // KPP - 2D
        if (k==0)
            return sin(rho);
        else if (k==1)
            return cos(rho);
    else if (typeF==2) {// Buckley-Leverett - 2D
        double deno = rho*rho + (1.0-rho)*(1.0-rho);
        if (k==0)
            return rho*rho/deno;
        else if (k==1)
            return rho*rho*(1.0-5.0*(1.0-rho)*(1.0-rho))/deno;
    }

}

double F_ccl(double rho, double rhobar, mfem::Vector mbar, mfem::Vector nbar, double alpha, int typeF)
{
   int k = mbar.Size();
   double val = 0.5*pow(rho - rhobar,2)/sigma_u;
   for (int i=0; i<k; i++){
       double fx = alpha* flux(rho, i, typeF);
       val += 0.5 * pow(mbar[i] - fx, 2) / (sigma_u + rho);
       val += 0.5 * pow(nbar[i] - fx, 2) / (sigma_u + rho);
       val -= 0.5 * fx * fx / rho;
   }
   return val;
}

double G(double rho)
{
   return 0.5*dE(rho)*dE(rho)*V2(rho);
}

double F(double rho, double rhobar, double mbar2, double nbar2, double sbar2)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u +0.5*mbar2/(sigma_u + V1(rho))
                                 + 0.5*nbar2/(sigma_u + V3(rho))
                                 + 0.5*sbar2/(sigma_u + V2(rho))
                                 + beta*beta*G(rho);
}

double F_sbp(double rho, double rhobar, double mbar2, double nbar2)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u +0.5*mbar2/(sigma_u + V1(rho))
                                 + 0.5*nbar2/(sigma_u + V3(rho));
}

double F_brain(double rho, double rhobar, double mbar2, double nbar2, double sbar2)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u +0.5*mbar2/(sigma_u + V1(rho))
                                 + 0.5*nbar2/(sigma_u + V3(rho))
                                 + 0.5*sbar2/(sigma_u + V2(rho)) ;
}

double F_brain_mono(double rho, double rhobar, double mbar2, double nbar2, double sbar2,
    double c0, double r0)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u +0.5*mbar2/(sigma_u + V1(rho))
                                 + 0.5*nbar2/(sigma_u + V3(rho))
                                 + 0.5*sbar2/(sigma_u + V2(rho)) 
                                 + c0 * pow(rho-r0, 2);
                                 //+ c0 * KL(rho, r0);
}

double Fb(double rho, double rhobar)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u + beta*E(rho);
}

// ho for high order flag
// c0 for KL regularizer
// rhoB & cB for blocking
double F_drop(double rho, double rhobar, double mbar2, mfem::Vector nbar, 
    double sbar2, double pbar, mfem::Vector qbar, bool ho, double c0, double rhoB, double cB)
{
    double de = dE_drop(rho);
    double d2e = d2E_drop(rho);
    double qnbar2 = 0.0;
    for (int d= 0; d < qbar.Size(); d++)
      qnbar2 += (qbar[d]+d2e*nbar[d])*(qbar[d]+d2e*nbar[d]);
    return pow(rho - rhobar,2)/sigma_u +mbar2/(sigma_u + V1(rho)) \
      + sbar2/(V2(rho)+sigma_u)
      + ho*beta*beta*V2(rho)*pow(pbar+de,2)/(1+V2(rho)*sigma_u*beta*beta)
      + ho*pow(beta/gma,2)*V1(rho)*qnbar2/(1+sigma_u*beta*beta/gma/gma*V1(rho)*
          (1+d2e*d2e))
      + c0*KL(rho, 1.0) 
      + cB * rhoB;
}

double KL(double rho, double rho0){
  return rho*log(rho/rho0);
}

double Fb_drop(double rho, double rhobar)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u + beta*E_drop(rho);
}

double Fb_drop(double rho, double rhobar, double rho0, double c0)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u + beta*E_drop(rho) 
     + c0 * KL(rho, rho0);
}

double F_drop0(double rho, double rhobar, double mbar2, double sbar2)
{
   return 0.5*pow(rho - rhobar,2)/sigma_u +0.5*mbar2/(sigma_u + V1(rho))
                                 + 0.5*sbar2/(sigma_u + V2(rho));
}
