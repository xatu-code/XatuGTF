#pragma once
#include "xatu/ConfigurationCRYSTAL_MPI.hpp"
#include "xatu/ConfigurationGTF.hpp"

namespace xatu {

/**
 * The ConfigurationGTF class is used to parse the file containing both the SCF and the AUXILIARY Gaussian basis sets. 
 * This class is intended to be used with the basis sets file in addition to the .outp file from the 
 * CRYSTAL code (see ConfigurationCRYSTAL.cpp for details on the latter). The former must be given in CRYSTAL 
 * format and contain both the basis employed in the self-consistent DFT calculation (under a line with the 
 * string: SCF BASIS) in addition to a larger auxiliary basis employed for fitting the density (under a line 
 * with the string: AUXILIARY BASIS). The initial occupancies have no impact. Pseudo-potentials are ignored, 
 * but it is advisable to be consistent with the PP choices in the self-consistency.
 * Exclusive to the GAUSSIAN mode
 */
class ConfigurationGTF_MPI final : public ConfigurationGTF {

    public:
        ConfigurationGTF_MPI(const int procMPI_rank, const int procMPI_size, const int nspecies, const std::vector<int>& atomic_number_ordering, const std::string& bases_file);
        ConfigurationGTF_MPI(const ConfigurationCRYSTAL_MPI&, const int procMPI_rank, const int procMPI_size, const std::string& bases_file);
        
};

}