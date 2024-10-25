#include "xatu/ConfigurationExciton_MPI.hpp"

namespace xatu {

/**
 * File constructor for ConfigurationExciton. It extracts the relevant information from the exciton file.
 * @param exciton_file Name of file with the exciton configuration.
 */
ConfigurationExciton_MPI::ConfigurationExciton_MPI(const std::string& exciton_file, const int procMPI_rank, const int procMPI_size) {

    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            if(exciton_file.empty()){
                throw std::invalid_argument("ConfigurationBase: file must not be empty");
            }
            m_file.open(exciton_file.c_str());
            if(!m_file.is_open()){
                throw std::invalid_argument("ConfigurationBase: file does not exist");
            }

            parseContent();

            m_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

    if(mode == "tb"){
        checkContentCoherenceTB();
    }
    else if(mode == "gaussian"){
        checkContentCoherenceGTF();
    }

}

}