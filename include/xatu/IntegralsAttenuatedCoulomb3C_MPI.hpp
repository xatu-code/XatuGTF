#pragma once
#include <mpi.h>
#include "xatu/IntegralsAttenuatedCoulomb2C_MPI.hpp"

namespace xatu {

/**
 * The IntegralsAttenuatedCoulomb3C class is designed to compute the three-center attenuated Coulomb integrals in the mixed SCF 
 * and AUXILIARY basis sets. It is called with the attenuated Coulomb metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsAttenuatedCoulomb3C_MPI : public IntegralsAttenuatedCoulomb2C_MPI {

    public:
        IntegralsAttenuatedCoulomb3C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR2, const std::string& intName = ""); 

    private:
        // Method to compute the rectangular attenuated Coulomb matrices <P,0|erfc(wr)V_c|mu,R;mu',R'> in the mixed SCF and auxiliary basis sets for the 
        // first nR2 Bravais vectors R and R' (nR2^2 pairs of vectors). These first nR (at least, until the star of vectors is completed)
        // are generated with IntegralsBase::generateRlist. Each entry above a certain tolerance (10^-tol) is stored in an entry of a 
        // vector (of arrays) along with the corresponding indices: value,P,mu,mu',R,R'; in that order. The vector is saved in the 
        // att0C3Mat_intName.att0C3c file. The list of Bravais vectors (for a single index) in fractional coordinates is saved in the RlistFrac_intName.att0C3c file
        void AttenuatedCoulomb3Cfun(const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR2, const std::string& intName);

};

}