#pragma once
#include "xatu/IntegralsBase.hpp"
#include "xatu/asa239.hpp"

namespace xatu {

/**
 * The IntegralsCoulomb2C class is designed to compute and store the two-center Coulomb integrals in the AUXILIARY basis set. 
 * It is called irrespective of the metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsCoulomb2C : public virtual IntegralsBase {

    protected:
        IntegralsCoulomb2C() = default;
    public:
        IntegralsCoulomb2C(const IntegralsBase&, const int tol, const std::vector<int32_t>& nRi, const std::string& intName = ""); 

    private:
        // Method to compute the Coulomb matrices in the auxiliary basis (<P,0|V_c|P',R>) for a set of Bravais vectors whose fractional
        // components span the number of values given in the vector nRi. This construction of R vectors is not by norm as in the other 
        // integrals due to the conditional convergence of Coulomb integrals, and it must be chosen so that nRi[n] is a multiple of nki[n].
        // Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
        // indices: value,mu,mu',R,; in that order. The vector is saved in the C2Mat_intName.C2c file, and the list of Bravais 
        // vectors in fractional coordinates is saved in the RlistFrac_intName.C2c file. Only the lower triangle of each R-matrix is stored; 
        // the upper triangle is given by hermiticity in the k-matrix
        void Coulomb2Cfun(const int tol, const std::vector<int32_t>& nRi, const std::string& intName);

    protected:
        // Analogous to Efun in the parent class IntegralsBase, but restricted to i'=0 and setting PA=0. These are the 
        // expansion coefficients of a single (cartesian) GTF in Hermite Gaussians with the same exponent and center
        arma::colvec Efun_single(const int i, const double p);
        // Method to compute and return the Boys function F_{n}(arg) = \int_{0}^{1}t^{2n}exp(-arg*t^2)dt. It is computed
        // with the lower incomplete Gamma function as: F_{n}(arg) = Gamma(n+0.5)*IncGamma(n+0.5,arg)/(2*arg^(n+0.5)), see (9.8.20)-Helgaker
        double Boysfun(const int n, const double arg);
        // Method to compute and return the auxiliary Hermite Coulomb integral R^{n}_{0,0,0} = (-2p)^n *F_{n}(arg), see (9.9.14)-Helgaker
        double Rn000(const int n, const double p, const double arg);
        // Method to compute and return the Hermite Coulomb integral R^{0}_{t,u,v}(r,(X,Y,Z)), see (9.9.9)-Helgaker. 
        // Restricted to (t + u + v) <= 12
        double HermiteCoulomb(const int t, const int u, const int v, const double p, const double X, const double Y, const double Z);
        // Compute the (normalized) vector with the charges of each orbital (fixed l,m) in the AUX basis, and save it to charges_intName.chg
        void computeCharges(const std::string& intName);

};

}