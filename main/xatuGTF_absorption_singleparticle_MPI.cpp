// #define ARMA_ALLOW_FAKE_GCC
#include <mpi.h>
#include <tclap/CmdLine.h>
#include "xatu.hpp"

// Based on the Delta function (real part of i*Lorentzian), computes only the first ndim components in the diagonal of the absorption tensor.
// For the off-diagonal components (and also for the imaginary parts of the diagonal components), the full Lorentzian version has to be used.
// Prefactor included: pi*spin_degeneracy/(Nk*V*omega)
int main(int argc, char* argv[]){

    int procMPI_rank, procMPI_size;

    MPI_Init (&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &procMPI_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &procMPI_size);
    if(procMPI_rank == 0){
        xatu::printHeaderGTFprovisional();
        std::cout << std::endl;
        xatu::printParallelizationMPI(procMPI_size);
        std::cout << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
        std::cout << "|                  POST BSE: ABSORPTION (SINGLE-PARTICLE)                   |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/hBN_HSE06.outp";
    int ncells = 179;                               //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string intName = "NEW_def2-TZVPPD-RIFIT";  //For the dipole integrals
    // Absorption parameters
    std::vector<int32_t> nki = {100,100};  //Number of k-points per G_{i} direction (Monkhorst-Pack grid) to discretize the BZ
    double scissor = 0.;                 //Rigid upwards traslation of the conduction bands, in eV
    double broadening = 0.05;            //Gaussian broadening for the Delta funcion, in eV
    double omega0 = 3.0;                 //First frequency value, in eV
    double omega1 = 8.0;                 //Last frequency value, in eV
    uint32_t nomega = 751;               //Number of points in the frequency grid, between omega0 and omega1
    std::string savefile = "BSEabsor";   //Result will be saved in file Results/4-Absorption/savefile.absorSP
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ResultGTF_MPI ResultGTF(CRYSTALconfig, procMPI_rank, procMPI_size, intName);

    // Prepare absorption-related quantities
    scissor *= EV2HARTREE;     //convert the input scissor correction from eV to Hartree
    broadening *= EV2HARTREE;  //convert the input broadening from eV to Hartree
    omega0 *= EV2HARTREE;      //convert the input omega0 from eV to Hartree
    omega1 *= EV2HARTREE;      //convert the input omega1 from eV to Hartree
    arma::colvec omegaDom = arma::linspace<arma::colvec>(omega0, omega1, nomega);  //1D grid of frequencies, in eV
    double spin_degeneracy_fac = (!ResultGTF.MAGNETIC_FLAG && !ResultGTF.SOC_FLAG)? 2. : 1.;

    // Generate and distribute k-points among MPI processes
    std::vector<int32_t> dummy_vec = {2};
    ResultGTF.initializekGrids(nki, dummy_vec);             // From now on, the "BSE grid" in ResultGTF is actually the grid given as input with the nki variable
    uint32_t nk_per_proc = ResultGTF.nkBSE / procMPI_size;  // number of k-points first distributed to each MPI process
    int32_t nk_remainder = ResultGTF.nkBSE % procMPI_size;  // remaining k-points first after first distribution
    uint p1 = std::min(nk_remainder, procMPI_rank);
    uint p2 = std::min(nk_remainder, procMPI_rank + 1);
    if(nk_per_proc == 0){
        std::cout << "There are idle MPI processes. Reduce the number to no more than " << ResultGTF.nkBSE << std::endl;
        throw std::invalid_argument("");
    }
    if(procMPI_rank == 0){
        std::cout << "k-points per process (max): " << nk_per_proc + p2 << std::endl;
    }

    // Compute the single-particle absorption by integrating over the BZ
    arma::mat sigma_diag = arma::zeros<arma::mat>(nomega, ResultGTF.ndim); //store the (real) diagonal sigma components resolved in omega, concatenated as (XX,YY,ZZ) until ndim
    for(uint32_t kind = procMPI_rank*nk_per_proc + p1; kind < (procMPI_rank + 1)*nk_per_proc + p2; kind++){
        arma::colvec k = ResultGTF.kpointsBSE.col(kind);
        arma::colvec sp_energies;
        arma::cx_mat vk = ResultGTF.velocities_vc(k, sp_energies, scissor);
        arma::cx_mat vkX = ( vk.cols(0, ResultGTF.ncbands - 1) ).st();
        arma::cx_mat vkY, vkZ;
        if(ResultGTF.ndim >= 2){
            vkY = ( vk.cols(ResultGTF.ncbands, 2*ResultGTF.ncbands - 1) ).st();
            if(ResultGTF.ndim == 3){
                vkZ = ( vk.cols(2*ResultGTF.ncbands, 3*ResultGTF.ncbands - 1) ).st();
            }
        }

        arma::rowvec Ev = (sp_energies.subvec(0, ResultGTF.highestValenceBand)).t();
        arma::colvec Ec = sp_energies.subvec(ResultGTF.filling, ResultGTF.norbitals - 1);
        arma::mat Ec_minus_Ev = arma::repmat(Ec, 1, ResultGTF.nvbands) - arma::repmat(Ev, ResultGTF.ncbands, 1); 

        for(uint32_t omega_ind = 0; omega_ind < nomega; omega_ind++){
            arma::mat delta_omega_arg = omegaDom(omega_ind)*arma::ones<arma::mat>(ResultGTF.ncbands, ResultGTF.nvbands) - Ec_minus_Ev;
            arma::mat delta_omega = arma::exp( - (delta_omega_arg % delta_omega_arg) * (0.5/(broadening*broadening)) );
            delta_omega /= Ec_minus_Ev;

            sigma_diag(omega_ind, 0) += arma::accu( arma::real( (vkX % delta_omega) % arma::conj(vkX) ) );
            if(ResultGTF.ndim >= 2){
                sigma_diag(omega_ind, 1) += arma::accu( arma::real( (vkY % delta_omega) % arma::conj(vkY) ) );
                if(ResultGTF.ndim == 3){
                    sigma_diag(omega_ind, 2) += arma::accu( arma::real( (vkZ % delta_omega) % arma::conj(vkZ) ) );
                }
            }
        }

    }
    sigma_diag *= PI/(std::sqrt(TWOPI)*broadening);
    sigma_diag *= spin_degeneracy_fac/(ResultGTF.nkBSE * ResultGTF.unitCellVolume*std::pow(ANG2AU,ResultGTF.ndim));

    // Reduce the k-summation 
    std::vector<double> sigma_diag_X = arma::conv_to<std::vector<double>>::from( sigma_diag.col(0) );
    std::vector<double> sigma_diag_Y, sigma_diag_Z;
    if(ResultGTF.ndim >= 2){
        sigma_diag_Y = arma::conv_to<std::vector<double>>::from( sigma_diag.col(1) );
        if(ResultGTF.ndim == 3){
            sigma_diag_Z = arma::conv_to<std::vector<double>>::from( sigma_diag.col(2) );
        }
    }
    std::vector<double> sigma_diag_X_tot(nomega); 
    std::vector<double> sigma_diag_Y_tot(nomega);
    std::vector<double> sigma_diag_Z_tot(nomega);

    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Reduce(&sigma_diag_X[0], &sigma_diag_X_tot[0], nomega, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ResultGTF.ndim >= 2){
        MPI_Reduce(&sigma_diag_Y[0], &sigma_diag_Y_tot[0], nomega, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ResultGTF.ndim == 3){
            MPI_Reduce(&sigma_diag_Z[0], &sigma_diag_Z_tot[0], nomega, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    if(procMPI_rank == 0){
        // Write absorption components to file
        // (first clear the corresponding file if already present, then write the header and finally the calculated values)
        savefile = "Results/4-Absorption/" + savefile + ".absorSP";
        std::string units_str = "e^{2}/hbar";
        std::string comp_dim_str = ", XX";
        if(ResultGTF.ndim >= 2){
            comp_dim_str += ", YY";
            if(ResultGTF.ndim == 3){
                comp_dim_str += ", ZZ";
            }
        }
        omegaDom /= EV2HARTREE;
        std::ofstream output_file0(savefile, std::ios::trunc);
        output_file0.close();
        std::ofstream output_file(savefile, std::ios::app);
        output_file << "SINGLE-PARTICLE ABSORPTION DIAGONAL COMPONENTS (DELTA FUNCTION)" << std::endl;
        output_file << "nk = " << ResultGTF.nkBSE << std::endl;
        output_file << "Fractional coordinates of Monkhorst-Pack grid: ";
        for(int d = 0; d < ResultGTF.ndim; d++){
            output_file << nki[d] << " ";
        } 
        output_file << std::endl;
        output_file << "Valence bands: All" << std::endl;
        output_file << "Conduction bands: All" << std::endl;
        output_file << "Broadening (eV): " << broadening/EV2HARTREE << ", Units: " << units_str << std::endl;
        output_file << "Frequency (eV)" + comp_dim_str << std::endl;
        output_file << "Number of frequency points: " << std::endl;
        output_file << nomega << std::endl;
        output_file.precision(12);
        output_file << std::scientific;
        for(uint32_t omega_ind = 0; omega_ind < nomega; omega_ind++){
            if(ResultGTF.ndim == 1){
                output_file << omegaDom(omega_ind) << "   " << sigma_diag_X_tot[omega_ind] << std::endl;
            }
            else if(ResultGTF.ndim == 2){
                output_file << omegaDom(omega_ind) << "   " << sigma_diag_X_tot[omega_ind] << " " << sigma_diag_Y_tot[omega_ind] << std::endl;
            }
            else{
                output_file << omegaDom(omega_ind) << "   " << sigma_diag_X_tot[omega_ind] << " " << 
                    sigma_diag_Y_tot[omega_ind] << " " << sigma_diag_Z_tot[omega_ind] << std::endl;
            }
        }
        output_file.close();
        std::cout << "Done! Diagonal absorption components stored in " << savefile << std::endl;
    }
    MPI_Barrier (MPI_COMM_WORLD);


    MPI_Finalize();
    return 0;

}