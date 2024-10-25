#pragma once
#include <mpi.h>
#include "xatu/IntegralsCoulomb2C.hpp"

namespace xatu {

/**
 * The IntegralsCoulomb2C class is designed to compute and store the two-center Coulomb integrals in the AUXILIARY basis set. 
 * It is called irrespective of the metric. Exclusive to the GAUSSIAN mode
 */
class IntegralsCoulomb2C_MPI : public virtual IntegralsCoulomb2C {

    public:
        IntegralsCoulomb2C_MPI(const IntegralsBase&, const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& nRi, const std::string& intName = ""); 

    private:
        // Method to compute the Coulomb matrices in the auxiliary basis (<P,0|V_c|P',R>) for a set of Bravais vectors whose fractional
        // components span the number of values given in the vector nRi. This construction of R vectors is not by norm as in the other 
        // integrals due to the conditional convergence of Coulomb integrals, and it must be chosen so that nRi[n] is a multiple of nki[n].
        // Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
        // indices: value,mu,mu',R,; in that order. The vector is saved in the C2Mat_intName.C2c file, and the list of Bravais 
        // vectors in fractional coordinates is saved in the RlistFrac_intName.C2c file. Only the lower triangle of each R-matrix is stored; 
        // the upper triangle is given by hermiticity in the k-matrix
        void Coulomb2Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& nRi, const std::string& intName);

};

}