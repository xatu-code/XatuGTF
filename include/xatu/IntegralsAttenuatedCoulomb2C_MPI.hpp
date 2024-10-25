#pragma once
#include <mpi.h>
#include "xatu/IntegralsCoulomb2C.hpp"

namespace xatu {

/**
 * The IntegralsAttenuatedCoulomb2C class is designed to compute and store the two-center attenuated Coulomb integrals in the AUXILIARY basis set. 
 * It is called with the attenuated Coulomb metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsAttenuatedCoulomb2C_MPI : public virtual IntegralsCoulomb2C {

    protected:
        IntegralsAttenuatedCoulomb2C_MPI() = default;
    public:
        IntegralsAttenuatedCoulomb2C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR, const std::string& intName = ""); 

    private:
        // Method to compute the attenuated Coulomb matrices in the auxiliary basis (<P,0|erfc(wr)V_c|P',R>) for the first nR Bravais vectors R. 
        // These first nR (at least, until the star of vectors is completed) are generated with Lattice::generateRlist.
        // Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
        // indices: value,mu,mu',R,; in that order. The vector is saved in the att0C2Mat_intName.att0C2c file, and the list of Bravais 
        // vectors in fractional coordinates is saved in the RlistFrac_intName.att0C2c file. Only the lower triangle of each R-matrix is stored; 
        // the upper triangle is given by hermiticity in the k-matrix
        void AttenuatedCoulomb2Cfun(const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR, const std::string& intName);

};

}