#pragma once
#include <mpi.h>
#include "xatu/ConfigurationCRYSTAL.hpp"

namespace xatu {

/**
 * The ConfigurationCRYSTAL class is used to parse CRYSTAL's .outp files, which can be used both 
 * in the TB and GAUSSIAN modes. This file is a post-scf processing which must contain H(R) and S(R).
 * This class will remove the unpaired Bravais vectors from Rlist and the Hamiltonian and overlap matrices.
 */
class ConfigurationCRYSTAL_MPI final : public ConfigurationCRYSTAL {

    public:
        ConfigurationCRYSTAL_MPI(const std::string& outp_file, const int procMPI_rank, const int procMPI_size, const int ncells, const bool isGTFmode);

};

}
