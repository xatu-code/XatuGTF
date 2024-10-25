#pragma once
#include <stdlib.h>
#include <omp.h>
#include "xatu/ConfigurationCRYSTAL.hpp"
#include "xatu/Lattice.hpp"

namespace xatu {

/**
 * The System class is an abstract class that contains the information, common to both TB and GAUSSIAN modes, relative to the 
 * system where we want to compute the exciton spectrum. It is defined as a sub-class of Lattice
*/
class System : public Lattice {

    public: // The attributes below do not have a protected counterpart
        // Imaginary unit i
        static constexpr std::complex<double> imag {0., 1.};
        // Hamiltonian or Fock matrices stored as a cube, where each slice represents an element R in Rlist (in the same ordering) 
        // and contains the matrix H(R) 
        const arma::cx_cube* ptr_hamiltonianMatrices;
        // Substitutes of hamiltonianMatrices under magnetism but no SOC. Alpha (beta) refer to up-up (down-down, respectively) spin blocks
        const arma::cx_cube* ptr_alphaMatrices; 
        const arma::cx_cube* ptr_betaMatrices;
        // Overlap stored as a cube, where each slice represents an element R in Rlist (in the same ordering) 
        // and contains the matrix S(R) 
        const arma::cx_cube* ptr_overlapMatrices;
    
    protected:
        // String labelling the system
        std::string systemName;
        // Number of filled bands, considering that each band holds 2 electrons at each k-point for spin-independent Hamiltonians
        int filling_;
        // Index of the highest occupied band, starting at 0. The system is assumed to be non-metallic
        int highestValenceBand_;
        // Number of unit cells for which the Hamiltonian (and possibly overlap) matrices are stored 
        int ncells_;
        // Single-particle matrix dimension, which equals the number of orbitals (with fixed l,m in the SCF basis for Gaussian mode)
        uint32_t norbitals_; 
    
    public:  // Const references to attributes (read-only)
        const int& filling            = filling_;
        const int& highestValenceBand = highestValenceBand_;
        const int& ncells             = ncells_;
        const uint32_t& norbitals     = norbitals_;

    protected:
        System(); 
        System(const ConfigurationSystem&);
    public:
        System(const System&) = default; 

    public:
        void setSystemName(const std::string& systemName);
        virtual void solveBands(const arma::colvec&, arma::colvec& eigval, arma::cx_mat& eigvec) = 0;
        void printBands(const std::string& kpointsfile);
        arma::mat generateBSEgrid(const std::vector<int32_t>& nki, const bool fractionalCoords = false);
          
};

}
