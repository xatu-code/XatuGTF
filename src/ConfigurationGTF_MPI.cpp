#include "xatu/ConfigurationGTF_MPI.hpp"


namespace xatu {

/**
 * File constructor for ConfigurationGTF from a full ConfigurationCRYSTAL object. It extracts the relevant information 
 * (exponents and contraction coefficients per shell per atomic species) from the basis sets file and stores it as attributes.
 * @param CRYSTALconfig ConfigurationCRYSTAL object.
 * @param bases_file Name of the file containing both Gaussian basis sets.
 */
ConfigurationGTF_MPI::ConfigurationGTF_MPI(const ConfigurationCRYSTAL_MPI& CRYSTALconfig, const int procMPI_rank, const int procMPI_size, const std::string& bases_file) {

    this->nspecies_ = CRYSTALconfig.nspecies;
    this->atomic_number_ordering_ = CRYSTALconfig.atomic_number_ordering;

    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            if(bases_file.empty()){
                throw std::invalid_argument("ConfigurationGTF_MPI: file must not be empty");
            }
            m_file.open(bases_file.c_str());
            if(!m_file.is_open()){
                throw std::invalid_argument("ConfigurationGTF_MPI: file does not exist");
            }

            parseContent();

            m_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

}

/**
 * File constructor for ConfigurationGTF just from the relevant attributes of an external ConfigurationCRYSTAL objetct. 
 * @param nspecies Number of chemical species in the unit cell
 * @param atomic_number_ordering Vector with the atomic number (+200 if pseudo-potential) of each chemical species, in the ordering displayed in the .outp file
 * @param bases_file Name of the file containing both Gaussian basis sets.
 */
ConfigurationGTF_MPI::ConfigurationGTF_MPI(const int procMPI_rank, const int procMPI_size, const int nspecies, const std::vector<int>& atomic_number_ordering, const std::string& bases_file) {

    this->nspecies_ = nspecies;
    this->atomic_number_ordering_ = atomic_number_ordering;
    
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            if(bases_file.empty()){
                throw std::invalid_argument("ConfigurationGTF_MPI: file must not be empty");
            }
            m_file.open(bases_file.c_str());
            if(!m_file.is_open()){
                throw std::invalid_argument("ConfigurationGTF_MPI: file does not exist");
            }

            parseContent();

            m_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }
    
}

}

