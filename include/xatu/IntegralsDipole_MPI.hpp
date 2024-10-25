#pragma once
#include <mpi.h>
#include "xatu/IntegralsDipole.hpp"

namespace xatu {

/**
 * The IntegralsDipole class is designed to compute the dipole integrals <mu,0|r|mu',R>, r = x,y,z, in the SCF basis set. 
 * It must be called to compute optical properties (where the arbitrareness in the choice of origin is cancelled), such as absorption. 
 * Exclusive to the GAUSSIAN mode
 */
class IntegralsDipole_MPI : public virtual IntegralsDipole {

    protected:
        IntegralsDipole_MPI() = default;
    public:
        IntegralsDipole_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName = ""); 

    private:
        // Method to compute the dipole matrices <mu,0|r|mu',R>, r = x,y,z, in the SCF basis set for the first nR Bravais vectors R. 
        // These first nR (at least, until the star of vectors is completed) are generated with Lattice::generateRlist. Each triplet of entries
        // (x,y,z) where at least one element is above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the 
        // corresponding indices: value_x,value_y,value_z,mu,mu',R; in that order. The vector is saved in the dipoleMat_intName.dip file in ATOMIC 
        // UNITS, and the list of Bravais vectors in fractional coordinates is saved in the RlistFrac_intName.dip file. The whole R-matrices are 
        // stored (both triangles) 
        void dipolefun(const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName);

};

}