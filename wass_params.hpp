#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "mfem.hpp"

// Global parameters
extern double eps; // newton tolerance
extern double cgrtol; // cg relative tolerance
extern double cgatol; // cg absolute tolerance
extern int cgi;
extern int iterPnt; // print every iterPnt steps
extern int test_case; // SBP test case: 1, 2, or 3
extern double gma, beta; // FIXME: we always take gma==beta
extern int typeI, typeV1, typeV2; // typeV1 = 0: V1=rho
                                  // typeV2 = 0: V2= (rho-1)/log(rho)
                                  // typeI: initial data
extern double typeE; // typeE = 1: E = rho(log(rho)-1)
                     // typeE > 1: E = rho^typeE/(typeE-1)
extern double c2; // reaction strength
extern double sigma_phi, sigma_u; // PDHG parameters: always set to 1

constexpr int double_bits = std::numeric_limits<double>::digits;

double rho_ex(const mfem::Vector &x, double t); // space-time soln
double rho_0(const mfem::Vector &x); // initial

double rho_ccl_ex(const mfem::Vector &x, double t); // space-time soln for CCL
double rho_ccl_pde_ex(const mfem::Vector &x, double t); // space-time soln
double rho_sbp_ex(const mfem::Vector &x, double t); // space-time soln for SBP
double rho_sbp_surf_ex(const mfem::Vector &x, double t); // space-time soln for SBP SURF

double rho_obs(const mfem::Vector &x, double t); // space-time obstacle
double rho_drop_ex(const mfem::Vector &x, double t); // space-time soln for SBP
double rho_drop_target(const mfem::Vector &x); // space-time soln for SBP

double E   (double);
double dE  (double);
double d2E (double);
double E_drop  (double);
double dE_drop  (double);
double d2E_drop (double);
double G   (double);
double V1  (double);
double V2  (double);
double V3  (double);
double flux  (double, int, int);

// Brent solver functions
double F(double rho, double rhobar, double mbar2,double nbar2, double sbar2);
double Fb(double rho, double rhobar);
double KL(double rho, double rho0);

double F_sbp(double rho, double rhobar, double mbar2, double nbar2);
double F_ccl(double rho, double rhobar, mfem::Vector mbar, mfem::Vector nbar, double alpha, int typeF=0);

double F_brain(double rho, double rhobar, 
        double mbar2, double nbar2, double sbar2);
double F_brain_mono(double rho, double rhobar, 
        double mbar2, double nbar2, double sbar2, double c0, double r0);

// with obstacle
double F_drop(double rho, double rhobar, 
    double mbar2, mfem::Vector nbar, double sbar2, double pbar, mfem::Vector qbar,
    bool ho = true, double ci=0.0, double rhoB=0.0, double cB=0.0);

double Fb_drop(double rho, double rhobar, double rho0, double cb);
double Fb_drop(double rho, double rhobar);
double F_drop0(double rho, double rhobar, double mbar2, double sbar2);

#endif
