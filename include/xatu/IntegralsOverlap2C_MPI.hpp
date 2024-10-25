#pragma once
#include <mpi.h>
#include "xatu/IntegralsOverlap2C.hpp"

namespace xatu {

/**
 * The IntegralsOverlap2C class is designed to compute and store the two-center overlap integrals in the SCF or AUXILIARY basis set. 
 * Only the integrals in the AUX basis are needed with the OVERLAP METRIC. Exclusive to the GAUSSIAN mode
 */
class IntegralsOverlap2C_MPI : public virtual IntegralsOverlap2C {

    public:
        IntegralsOverlap2C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName = "", const bool basis_id = false); 

    private:
        // Method to compute the overlap matrices in the auxiliary (if basis_id == false) or SCF (if basis_id == true) basis 
        // (<P,0|P',R> or <mu,0|mu',R>) for the first nR Bravais vectors R. These first nR (at least, until the star of vectors is 
        // completed) are generated with Lattice::generateRlist. Each entry above a certain tolerance (10^-tol) is stored in an entry 
        // of a vector (of arrays) along with the corresponding indices: value,mu,mu',R; in that order. The vector is saved in the 
        // o2Mat_intName.o2c file, and the list of Bravais vectors in fractional coordinates is saved in the RlistFrac_intName.o2c file.
        // Only the lower triangle of each R-matrix is stored; the upper triangle is given by hermiticity in the k-matrix
        void overlap2Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName, const bool basis_id = false);

};

}