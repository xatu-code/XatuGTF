#pragma once
#include <numeric>
#include <iomanip>
#include <math.h>
#include <chrono>
#include "xatu/ConfigurationSystem.hpp"

#ifndef constants
#define PI 3.141592653589793
#define TWOPI 6.283185307179586
#endif

namespace xatu {

/** 
 *  The Lattice class serves as a base for the family of classes that manipulates information about reciprocal and direct space. 
 */
class Lattice {
    
    protected:
        // Lattice dimensionality (0,1,2,3)
        int ndim_;
        // Basis of Bravais vectors (R1,R2,R3) in Angstrom, stored by columns: (3,ndim)
        arma::mat Rbasis_;
        // Unit cell volume in Angstrom^ndim
        double unitCellVolume_;
        // Basis of reciprocal vectors (G1,G2,G3) in Angstrom^-1, stored by columns: (3,ndim)
        arma::mat Gbasis_;
        
    public:  // Const references to attributes (read-only)
        const int& ndim = ndim_;
        const arma::mat& Rbasis = Rbasis_;
        const double& unitCellVolume = unitCellVolume_;
        const arma::mat& Gbasis = Gbasis_;

    protected:
        Lattice() = default;
    public:
        Lattice(const ConfigurationSystem&);
        Lattice(const Lattice&) = default; 
        virtual ~Lattice() = default;

    public: 
        // Rearrange vector ni, coming from the exciton input (flexible), assigning a value for each spatial component, duplicating values if necessary
        void unify_ni(const std::vector<int32_t>& ni, int32_t& n1, int32_t& n2, int32_t& n3);
        // Method to generate a kronecker-like list of integer combinations, to be used with Bravais or reciprocal lattice vectors
        arma::mat generateCombinations(const std::vector<int32_t>& ni, const bool centered = false);
        // Compute a Monkhorst-Pack grid in the interval [0 G1)x...x[0 Gn_dim), and return the k-points by columns in arma::mat (3,nk).
        // If fractionalCoords = false, the default units are Angstrom^-1, as for Gbasis
        arma::mat gridMonkhorstPack(const std::vector<int32_t>& nki, const bool containsGamma = true, const bool fractionalCoords = false);
        // Method to create the matrix of the first nG (at least) 3-component reciprocal lattice vectors, stored by columns and ordered by ascending norm.
        // Default units: Angstrom^-1. The number of returned vectors is at least nG because full stars are given. 
        arma::mat generateGlist(const uint32_t nG, arma::mat& combs, const int procMPI_rank = 0);
        // Method to create the matrix of the first nR (at least) 3-component Bravais vectors, stored by columns and ordered by ascending norm.
        // Default units: Angstrom. The number of returned vectors is at least nR because full stars are given. 
        arma::mat generateRlist(const uint32_t nR, arma::mat& combs, const std::string& IntegralType, const int procMPI_rank = 0);
        // Method to create the matrix of the first nG (at least) 3-component reciprocal vectors in a supercell defined by scale factors in scalei
        // (each component corresponding a reciprocal basis vector). The generated vectors are stored by columns and ordered by ascending norm.
        // Default units: Angstrom^-1. The number of returned vectors is at least nG because full stars are given.  
        arma::mat generateGlist_supercell(const uint32_t nG, const std::vector<int32_t>& scalei, const int procMPI_rank = 0);
        // Analogous to generateGlist_supercell, but only one of each (G_{n},-G_{n}) pair is included, and G = 0 is excluded. 
        arma::mat generateGlist_supercell_half(const uint32_t nG, const std::vector<int32_t>& scalei, const int procMPI_rank = 0);
        // Method to create the matrix of the first nR (at least) 3-component Bravais vectors in a supercell defined by scale factors in scalei
        // (each component corresponding a Bravais basis vector). The generated vectors are stored by columns and ordered by ascending norm.
        // Default units: Angstrom. The number of returned vectors is at least nR because full stars are given. 
        arma::mat generateRlist_supercell(const uint32_t nR, const std::vector<int32_t>& scalei, const int procMPI_rank = 0);
        // Method to create a list of Bravais vectors, stored by columns, whose fractional coordinates ({R1,R2,R3} basis) are spanned by an 
        // input vector of maximum values. Default units: Angstrom.
        arma::mat generateRlist_fixed(const std::vector<int32_t>& nRi, arma::mat& combs, const std::string& IntegralType, const int procMPI_rank = 0);
        // Returns a map where each entry is the index of the vector in the input generated_Rlist (stores vector by columns)
        // opposite to the vector whose index is the corresponding map's key. 
        std::map<uint32_t,uint32_t> generateRlistOpposite(const arma::mat& generated_Rlist);
        
    protected:
        // Method to compute the unit cell volume, area or length (depending on the lattice dimensionality)
        void computeUnitCellVolume();
        // Compute the reciprocal lattice vectors {G_1,..,G_ndim} and return them by columns in arma::mat (3,ndim)
        void calculateGbasis();

};

}
