// #define ARMA_ALLOW_FAKE_GCC
#include <mpi.h>
#include <omp.h>
#include "xatu.hpp"

int main(int argc, char* argv[]){

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string diag_file = "custom_100x100";     //Diagonal BSE Hamiltonian file, excluding exchange
    std::string offdiag_file = "custom";          //Off-diagonal BSE Hamiltonian file, excluding exchange
    std::string X_file = "custom_100x100";        //Exchange file, irrelevant if exchange = triplet
    std::string exchange = "singlet";          //Exchange factor: "singlet" (x2), "SOC" (x1), "triplet" (x0). For triplet, exchange is not read
    std::string savefile = "BSE_singlet";      //Results will be saved in file Results/3-Excitons/savefile.energ & savefile.eigvec
    bool writeStates = true;                   //Store eigenvectors or not
    arma::uvec cols_eigvec = arma::regspace<arma::uvec>(0,3199);  //Indices of the eigenvectors to store, irrelevant if writeStates = false
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::printHeaderGTFprovisional();
    std::cout << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                              BSE DIAGONALIZE                              |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;

    diag_file    = "Results/2-BSEHamiltonian/" + diag_file + ".diag";
    offdiag_file = "Results/2-BSEHamiltonian/" + offdiag_file + ".offdiag";
    X_file       = "Results/2-BSEHamiltonian/" + X_file + ".exch";

    constexpr std::complex<double> imag {0., 1.};
    uint64_t dimBSE_0, dimBSE_1, dimBSE_X;
    // Read pre-computed BSE Hamiltonian entries (diagonal blocks, no exchange)
    {
        if(diag_file.empty()){
            throw std::invalid_argument("File with diagonal BSE blocks must not be empty");
        }
        std::ifstream diag_ifstream;
        diag_ifstream.open(diag_file.c_str());
        if(!diag_ifstream.is_open()){
            throw std::invalid_argument("File with diagonal BSE blocks does not exist");
        }
        std::string line;
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> dimBSE_0;
        diag_ifstream.close();
    }
    arma::cx_mat HBSE(dimBSE_0, dimBSE_0);

    {
        std::ifstream diag_ifstream;
        diag_ifstream.open(diag_file.c_str());
        std::string line;
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::getline(diag_ifstream, line);
        std::istringstream iss0(line);
        uint64_t nvalues_diag;
        iss0 >> nvalues_diag;

        double diag_Re, diag_Im;
        uint32_t i_BSE, j_BSE;
        for(uint64_t ent_diag = 0; ent_diag < nvalues_diag; ent_diag++){
            std::getline(diag_ifstream, line);
            std::istringstream iss1(line);
            iss1 >> diag_Re >> diag_Im >> i_BSE >> j_BSE;
            HBSE(i_BSE, j_BSE) = diag_Re + imag*diag_Im;
            if(i_BSE > j_BSE){
                HBSE(j_BSE, i_BSE) = diag_Re - imag*diag_Im;
            }
        }
        diag_ifstream.close();
    }

    // Read pre-computed BSE Hamiltonian entries (off-diagonal blocks, no exchange)
    {
        if(offdiag_file.empty()){
            throw std::invalid_argument("File with off-diagonal BSE blocks must not be empty");
        }
        std::ifstream offdiag_ifstream;
        offdiag_ifstream.open(offdiag_file.c_str());
        if(!offdiag_ifstream.is_open()){
            throw std::invalid_argument("File with off-diagonal BSE blocks does not exist");
        }
        std::string line;
        std::getline(offdiag_ifstream, line);
        std::getline(offdiag_ifstream, line);
        std::getline(offdiag_ifstream, line);
        std::getline(offdiag_ifstream, line);
        std::getline(offdiag_ifstream, line);
        std::getline(offdiag_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> dimBSE_1;
        if(dimBSE_0 != dimBSE_1){
            throw std::invalid_argument("ERROR: Dimensions of diagonal and off-diagonal Hamiltonian blocks do not match");
        }

        uint64_t nvalues_offdiag;
        std::getline(offdiag_ifstream, line);
        std::istringstream iss0b(line);
        iss0b >> nvalues_offdiag;
        double offdiag_Re, offdiag_Im;
        uint32_t ioff_BSE, joff_BSE;
        for(uint64_t ent_offdiag = 0; ent_offdiag < nvalues_offdiag; ent_offdiag++){
            std::getline(offdiag_ifstream, line);
            std::istringstream iss1(line);
            iss1 >> offdiag_Re >> offdiag_Im >> ioff_BSE >> joff_BSE;
            HBSE(ioff_BSE, joff_BSE) = offdiag_Re + imag*offdiag_Im;
            HBSE(joff_BSE, ioff_BSE) = offdiag_Re - imag*offdiag_Im;
        }
        offdiag_ifstream.close();
    }

    // Read pre-computed exchange entries (lower triangle)
    double Xfactor = 0.0;
    if(exchange == "singlet"){
        Xfactor = 2.0;
    } 
    else if(exchange == "SOC"){
        Xfactor = 1.0;
    }
    else if(exchange == "triplet"){
        Xfactor = 0.0;
    }

    if(exchange != "triplet"){
        if(X_file.empty()){
            throw std::invalid_argument("File with exchange entries must not be empty (unless triplet is selected)");
        }
        std::ifstream X_ifstream;
        X_ifstream.open(X_file.c_str());
        if(!X_ifstream.is_open()){
            throw std::invalid_argument("File with exchange entries does not exist. Either compute it, or select triplet");
        }
        std::string line;
        std::getline(X_ifstream, line);
        std::getline(X_ifstream, line);
        std::getline(X_ifstream, line);
        std::getline(X_ifstream, line);
        std::getline(X_ifstream, line);
        std::getline(X_ifstream, line);
        std::istringstream iss0(line);
        iss0 >> dimBSE_X;
        if(dimBSE_0 != dimBSE_X){
            throw std::invalid_argument("ERROR: Dimensions of direct and exchange Hamiltonian parts do not match");
        }

        uint64_t nvalues_X;
        std::getline(X_ifstream, line);
        std::istringstream iss0b(line);
        iss0b >> nvalues_X;
        double X_Re, X_Im;
        uint32_t iX_BSE, jX_BSE;
        for(uint64_t ent_X = 0; ent_X < nvalues_X; ent_X++){
            std::getline(X_ifstream, line);
            std::istringstream iss1(line);
            iss1 >> X_Re >> X_Im >> iX_BSE >> jX_BSE;
            HBSE(iX_BSE, jX_BSE) += Xfactor*(X_Re + imag*X_Im);
            if(iX_BSE > jX_BSE){
                HBSE(jX_BSE, iX_BSE) += Xfactor*(X_Re - imag*X_Im);
            }
        }
        X_ifstream.close();
    }

    // Diagonalize the BSE Hamiltonian
    arma::colvec energies_BSE;
    savefile = "Results/3-Excitons/" + savefile;
    if(writeStates){
        arma::cx_mat eigvec_BSE;
        std::cout << "Diagonalizing BSE Hamiltonian ... " << std::flush;
        arma::eig_sym(energies_BSE, eigvec_BSE, HBSE);
        std::cout << "Done!" << std::endl;

        energies_BSE = energies_BSE/EV2HARTREE;
        std::cout << "Storing energies..." << std::flush;
        energies_BSE.save(savefile + ".energ", arma::arma_ascii);
        std::cout << "Done! Energies (eV) stored in " << savefile << ".energ file." << std::endl;

        arma::cx_mat eigvec_BSE_tosave = eigvec_BSE.cols(cols_eigvec);
        std::cout << "Storing exciton coefficients..." << std::flush;
        eigvec_BSE_tosave.save(savefile + ".eigvec", arma::arma_ascii);
        std::cout << "Done! Coefficients stored in " << savefile << ".eigvec file." << std::endl;
    }
    else {
        std::cout << "Diagonalizing BSE Hamiltonian ... " << std::flush;
        energies_BSE = arma::eig_sym(HBSE);
        std::cout << "Done!" << std::endl;

        energies_BSE = energies_BSE/EV2HARTREE;
        std::cout << "Storing energies..." << std::flush;
        energies_BSE.save(savefile + ".energ", arma::arma_ascii);
        std::cout << "Done! Energies (eV) stored in " << savefile << ".energ file." << std::endl;
    }

}
