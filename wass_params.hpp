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
extern double c2; // reaction strength
extern double sigma_phi, sigma_u; // PDHG parameters: always set to 1

constexpr int double_bits = std::numeric_limits<double>::digits;

double rho_drop_ex(const mfem::Vector &x, double t); // space-time soln for SBP
double rho_drop_target(const mfem::Vector &x); // space-time soln for SBP

double E_drop  (double);
double dE_drop  (double);
double d2E_drop (double);
double V1  (double);
double V2  (double);

// Brent solver functions
double KL(double rho, double rho0);

// JKO functional
double F_drop(double rho, double rhobar, double mbar2, double sbar2);

double F_drop(double rho, double rhobar, 
    double mbar2, mfem::Vector nbar, double sbar2, double pbar, mfem::Vector qbar,
    double ci=0.0);

double Fb_drop(double rho, double rhobar, double rho0, double cb);
double Fb_drop(double rho, double rhobar);
double F_drop0(double rho, double rhobar, double mbar2, double sbar2);

#endif
