#include "wass_params.hpp"

using namespace mfem;

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
  else if (test_case==34 || test_case==36 || test_case==38){
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
  else if (test_case==47){ // transport
    double rad2 = (x(0)-0.3)*(x(0)-0.3) + (x(1)-0.3)*(x(1)-0.3);
    double mu = 0.1;
    return eps + 2.0 /(1+exp(rad2/mu/mu-1));
  }
  else if (test_case==39) // JKO
    return 1.0 + 0.2*cos(2.0*M_PI*x(0))*cos(2.0*M_PI*x(1));
  else if (test_case==29) // JKO
    return 1.0 - 0.2*cos(2.0*M_PI*x(0));
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
  else if (test_case == 36){ // asymmetric merging I
    double fac = 1.0;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = 4.0* (x(0)-0.6)*(x(0)-0.6) + (x(1)-0.5)*(x(1)-0.5);
    double val = 0.5*H1*(w0*w0-rad2/0.01/0.01);
    if (val >0)
      return eps+val;
    else
      return eps;
  }
  else if (test_case == 38){ // asymmetric merging II
    double fac = 1.0;
    double H0 = 0.05/fac, H1 = 0.05/pow(fac,3);
    double w0 = 1.0/sqrt(3)/H0;
    double rad2 = 4.0* (x(0)-0.6)*(x(0)-0.6) + (x(1)-0.55)*(x(1)-0.55);
    double val = 0.5*H1*(w0*w0-rad2/0.01/0.01);
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
  else if (test_case==47){ // transport
    double rad2 = (x(0)-0.7)*(x(0)-0.7) + (x(1)-0.7)*(x(1)-0.7);
    double mu = 0.1;
    return eps + 2.0 /(1+exp(rad2/mu/mu-1));
  }
  else
    return 1.0 + 0.2*cos(2.0*M_PI*x(0))*cos(2*M_PI*x(1));
}

// droplet energy
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
    return pow(rho,3); 
}

// reaction mobility
double V2(double rho)
{
   return c2/(rho+0.1);
}

// for JKO
double F_drop(double rho, double rhobar, double mbar2, double sbar2)
{
    return pow(rho - rhobar,2)/sigma_u +mbar2/(sigma_u + V1(rho)) \
      + sbar2/(V2(rho)+sigma_u);
}

// c0 for KL regularizer
double F_drop(double rho, double rhobar, double mbar2, mfem::Vector nbar, 
    double sbar2, double pbar, mfem::Vector qbar, double c0)
{
    double de = dE_drop(rho);
    double d2e = d2E_drop(rho);
    double qnbar2 = 0.0;
    for (int d= 0; d < qbar.Size(); d++)
      qnbar2 += (qbar[d]+d2e*nbar[d])*(qbar[d]+d2e*nbar[d]);
    return pow(rho - rhobar,2)/sigma_u +mbar2/(sigma_u + V1(rho)) \
      + sbar2/(V2(rho)+sigma_u)
      + beta*beta*V2(rho)*pow(pbar+de,2)/(1+V2(rho)*sigma_u*beta*beta)
      + pow(beta/gma,2)*V1(rho)*qnbar2/(1+sigma_u*beta*beta/gma/gma*V1(rho)*
          (1+d2e*d2e))
      + c0 * rho * log(rho); // regularization term 
}

double KL(double rho, double rho0){
  return rho*(log(rho/rho0)-1.0);
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
