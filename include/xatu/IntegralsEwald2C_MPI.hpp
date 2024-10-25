#pragma once
#include <mpi.h>
#include "xatu/IntegralsEwald2C.hpp"

#ifndef constants
#define PISQRT_INV 0.564189583547756
#endif

namespace xatu {

/**
 * The IntegralsEwald2C class is designed to compute and store the two-center Coulomb integrals in the AUXILIARY basis set, computed with the Ewald substitution. 
 * It is called irrespective of the metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsEwald2C_MPI : public virtual IntegralsEwald2C {

    public:
        IntegralsEwald2C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const bool is_for_Dk = false, const std::string& intName = ""); 

    protected:
        // Method to compute the Ewald matrices in the auxiliary basis (<P,0|A|P',R>) for a set of Bravais vectors within a supercell defined by scale
        // factors scalei_supercell (conmensurate with the k-grid for the BSE). Both direct and reciprocal lattice sums must be performed in the Ewald
        // potential, with a minimum number of terms nR and nG (respectively), in a lattice where the supercell is the unit of periodicity.
        // Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
        // indices: value,mu,mu',R,; in that order. The vector is saved in the E2Mat_intName.E2c file, and the list of supercell Bravais 
        // vectors in fractional coordinates is saved in the RlistFrac_intName.E2c file. Only the lower triangle of each R-matrix is stored; 
        // the upper triangle is given by hermiticity in the k-matrix
        void Ewald2Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const bool is_for_Dk, const std::string& intName);

};

}