#include "xatu/ConfigurationCRYSTAL_MPI.hpp"

#define SOC_STRING "to_be_defined_for_crystal23"
#define MAGNETIC_STRING "UNRESTRICTED OPEN SHELL"

namespace xatu {

/**
 * File constructor for ConfigurationCRYSTAL. It extracts the relevant information from CRYSTAL's .outp file.
 * @details This class is intended to be used with .outp files from the CRYSTAL code.
 * Since orbitals in CRYSTAL extend over several unit cells, the Fock matrices that define the
 * Hamiltonian also cover several unit cells. Therefore, one can specify how many unit cells to take
 * for the actual exciton calculation. 
 * @param outp_file Name of the .outp from the CRYSTAL post-scf calculation.
 * @param ncells Number of unit cells for which the Hamiltonian and overlap matrices are read from file.
 * @param isGTFmode True (false) indicates GTF (TB, resp.) mode. It determines whether hamiltonianMatrices or alpha/betaMatrices are 
 *        used in the case of magnetism without SOC.
 */
ConfigurationCRYSTAL_MPI::ConfigurationCRYSTAL_MPI(const std::string& outp_file, const int procMPI_rank, const int procMPI_size, const int ncells, const bool isGTFmode) {

    this->ncells_ = ncells;
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            if(outp_file.empty()){
                throw std::invalid_argument("ConfigurationCRYSTAL_MPI: file must not be empty");
            }
            m_file.open(outp_file.c_str());
            if(!m_file.is_open()){
                throw std::invalid_argument("ConfigurationCRYSTAL_MPI: file does not exist");
            }

            parseContent();

            m_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

    Rlistsfun(procMPI_rank);

    if(!isGTFmode && MAGNETIC_FLAG && !SOC_FLAG){ // In the TB mode, build H(R) and S(R) matrices in spin space (in GTF mode, spin matrices are treated separately)
        arma::mat spinUpBlock = {{1, 0}, {0, 0}};
        arma::mat spinDownBlock = {{0, 0}, {0, 1}};
        arma::cx_cube newOverlapMatrices;
        for(int i = 0; i < ncells; i++){
            arma::cx_mat totalFockMatrix = arma::kron(this->alphaMatrices.slice(i), spinUpBlock) + arma::kron(this->betaMatrices.slice(i), spinDownBlock);
            this->hamiltonianMatrices_ = arma::join_slices(this->hamiltonianMatrices_, totalFockMatrix);
            arma::cx_mat totalOverlapMatrix = arma::kron(this->overlapMatrices.slice(i), arma::eye(2, 2));
            newOverlapMatrices = arma::join_slices(newOverlapMatrices, totalOverlapMatrix);
        }
        this->overlapMatrices_ = newOverlapMatrices;
    }

}

}