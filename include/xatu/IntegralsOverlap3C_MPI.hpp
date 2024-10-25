#pragma once
#include <mpi.h>
#include "xatu/IntegralsOverlap3C.hpp"

namespace xatu {

/**
 * The IntegralsOverlap3C class is designed to compute the three-center overlap integrals in the mixed SCF and AUXILIARY basis sets. 
 * It is called if and only if the OVERLAP METRIC is chosen. Exclusive to the GAUSSIAN mode
 */
class IntegralsOverlap3C_MPI : public IntegralsOverlap3C {

    public:
        IntegralsOverlap3C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR2, const std::string& intName = ""); 

    private:
        // Method to compute the rectangular overlap matrices <P,0|mu,R;mu',R'> in the mixed SCF and auxiliary basis sets for the 
        // first nR2 Bravais vectors R and R' (nR2^2 pairs of vectors). These first nR (at least, until the star of vectors is completed)
        // are generated with IntegralsBase::generateRlist. Each entry above a certain tolerance (10^-tol) is stored in an entry of a 
        // vector (of arrays) along with the corresponding indices: value,P,mu,mu',R,R'; in that order. The vector is saved in the 
        // o3Mat_intName.o3c file. The list of Bravais vectors (for a single index) in fractional coordinates is saved in the RlistFrac_intName.o3c file
        void overlap3Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR2, const std::string& intName);

};

}