#pragma once
#include <armadillo>
#include <mpi.h>
#include "xatu/ConfigurationExciton.hpp"


namespace xatu {

/**
 * The ConfigurationExciton class is a specialization of ConfigurationBase to parse exciton configuration files.
 * It is used in both TB and GAUSSIAN modes, and the data is stored in a different struct for each of them
 */
class ConfigurationExciton_MPI : public ConfigurationExciton{

    public:
        ConfigurationExciton_MPI(const std::string& exciton_file, const int procMPI_rank, const int procMPI_size);
    
};

}