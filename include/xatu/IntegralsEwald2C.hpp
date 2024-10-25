#pragma once
#include "xatu/IntegralsCoulomb2C.hpp"

#ifndef constants
#define PISQRT_INV 0.564189583547756
#endif

namespace xatu {

/**
 * The IntegralsEwald2C class is designed to compute and store the two-center Coulomb integrals in the AUXILIARY basis set, computed with the Ewald substitution. 
 * It is called irrespective of the metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsEwald2C : public virtual IntegralsCoulomb2C {

    protected:
        // Reference value for the arbitrary gamma parameter in the Ewald potential, in atomic units for length^-2
        double gamma0_;
        // Square root of gamma0
        double gamma0_sqrt_;
        // Matrix with the list of direct lattice vectors (by columns and in atomic units) to be included in the Ewald potential sums, i.e. in a given supercell
        arma::mat RlistAU_outer_;
        // Number of supercell direct vectors included in RlistAU_outer
        uint32_t nR_outer_;
        // Matrix with the list of reciprocal lattice vectors (by columns and in atomic units) to be included in the Ewald potentials sums, i.e. without opposites and in a given supercell
        arma::mat GlistAU_half_;
        // Number of supercell reciprocal vectors included in GlistAU_half
        uint32_t nG_;
        // Vector with the norm (sqrt) of the corresponding reciprocal lattice vector G in the list to be summed in the Ewald potentials (no opposites). In atomic units
        arma::rowvec GlistAU_half_norms_;
    public: 
        const double& gamma0 = gamma0_;
        const double& gamma0_sqrt = gamma0_sqrt_;
        const arma::mat& RlistAU_outer = RlistAU_outer_;
        const uint32_t& nR_outer = nR_outer_;
        const arma::mat& GlistAU_half = GlistAU_half_;
        const uint32_t& nG = nG_;
        const arma::rowvec& GlistAU_half_norms = GlistAU_half_norms_;

    protected:
        IntegralsEwald2C() = default;
    public:
        IntegralsEwald2C(const IntegralsBase&, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const std::string& intName = ""); 

    protected:
        // Method to set the gamma0 attribute depending on lattice dimensionality
        void setgamma0(const std::vector<int32_t>& scalei_supercell);
        // Method to compute the Ewald matrices in the auxiliary basis (<P,0|A|P',R>) for a set of Bravais vectors within a supercell defined by scale
        // factors scalei_supercell (conmensurate with the k-grid for the BSE). Both direct and reciprocal lattice sums must be performed in the Ewald
        // potential, with a minimum number of terms nR and nG (respectively), in a lattice where the supercell is the unit of periodicity.
        // Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
        // indices: value,mu,mu',R,; in that order. The vector is saved in the E2Mat_intName.E2c file, and the list of supercell Bravais 
        // vectors in fractional coordinates is saved in the RlistFrac_intName.E2c file. Only the lower triangle of each R-matrix is stored; 
        // the upper triangle is given by hermiticity in the k-matrix
        void Ewald2Cfun(const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const std::string& intName);
        // Method to compute the direct lattice contribution to the Ewald integrals
        double Ewald2Cdirect(const arma::colvec& coords_braket, const int t_tot, const int u_tot, const int v_tot, const double mu);
        // Method to compute the reciprocal lattice contribution to the Ewald integrals, which depends on the dimensionality of the system (2D or 3D)
        double Ewald2Creciprocal(const arma::colvec& coords_braket, const int t_tot, const int u_tot, const int v_tot, const double gamma_fac);
        // Method to evaluate the n-th derivative of the cosine function at arg
        double derivative_cos(const int n, const double arg);
        // Method to evaluate the n-th derivative with respect to (r_1z - r_2z) of the G = 0 term in the 2D Ewald potential 
        double derivative_G02Dz(const int v_tot, const double bz, const double b);
        // Method to evaluate the n-th derivative with respect to (r_1z - r_2z) of the term in the 2D Ewald potential that is summed over G
        double derivative_Gfinite2Dz(const int v_tot, const double a, const double bz, const double b_pow);

};

}