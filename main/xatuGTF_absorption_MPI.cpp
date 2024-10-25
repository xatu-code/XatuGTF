// #define ARMA_ALLOW_FAKE_GCC
#include <mpi.h>
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
        std::cout << "|                           POST BSE: ABSORPTION                            |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp";
    int ncells = 43;                               //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string exciton_file = "InputFiles/Ph_88x66_alpha.txt";
    std::string intName = "custom";                //For the dipole integrals
    std::string excName = "BSE_singlet";                   //For the exciton files (.energ & .eigvec)
    // Absorption parameters
    uint32_t nA = 3200;                    //Number of excitons included in the sum
    double scissor = 0.;                 //Rigid upwards traslation of the conduction bands, in eV
    double broadening = 0.05;            //Gaussian broadening for the Delta funcion, in eV
    double omega0 = 1.0;                 //First frequency value, in eV
    double omega1 = 4.5;                 //Last frequency value, in eV
    uint32_t nomega = 901;               //Number of points in the frequency grid, between omega0 and omega1
    std::string savefile = "BSE";   //Result will be saved in file Results/4-Absorption/savefile.absor
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationExciton_MPI Excitonconfig(exciton_file, procMPI_rank, procMPI_size);
    xatu::ResultGTF_MPI ResultGTF(Excitonconfig, CRYSTALconfig, procMPI_rank, procMPI_size, nA, intName, excName);

    // Prepare absorption-related quantities
    scissor *= EV2HARTREE;     //convert the input scissor correction from eV to Hartree
    broadening *= EV2HARTREE;  //convert the input broadening from eV to Hartree
    omega0 *= EV2HARTREE;      //convert the input omega0 from eV to Hartree
    omega1 *= EV2HARTREE;      //convert the input omega1 from eV to Hartree
    arma::colvec omegaDom = arma::linspace<arma::colvec>(omega0, omega1, nomega);  //1D grid of frequencies, in eV
    arma::rowvec Eexc_nA = ( ResultGTF.Eexc.subvec(0, nA - 1) ).t();
    arma::mat delta_meshgrid = arma::repmat(omegaDom, 1, nA) - arma::repmat(Eexc_nA, nomega, 1); //(m,n) entry is omega_{m}-Eexc_{n}
    delta_meshgrid = - (delta_meshgrid % delta_meshgrid) * (0.5/(broadening*broadening));
    delta_meshgrid = arma::exp(delta_meshgrid) * PI/(std::sqrt(TWOPI)*broadening);   //Gaussian approximation for the Delta, in grid form
    arma::rowvec Eexc_inv = 1./Eexc_nA;  
    // double spin_degeneracy_fac = (!ResultGTF.MAGNETIC_FLAG && !ResultGTF.SOC_FLAG)? 2. : 1.;
    double spin_degeneracy_fac = 1.;

    // Distribute k-points among MPI processes
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

    // Compute the contribution to V^{a} (~ oscillator strengths) from the set of k-points assigned to this MPI process
    arma::cx_rowvec Vk_exc_X = arma::zeros<arma::cx_rowvec>(nA);
    arma::cx_rowvec Vk_exc_Y = arma::zeros<arma::cx_rowvec>(nA);
    arma::cx_rowvec Vk_exc_Z = arma::zeros<arma::cx_rowvec>(nA);
    for(uint32_t kind = procMPI_rank*nk_per_proc + p1; kind < (procMPI_rank + 1)*nk_per_proc + p2; kind++){
        arma::colvec k = ResultGTF.kpointsBSE.col(kind);
        arma::colvec sp_energies;
        arma::cx_mat vk = ResultGTF.velocities_vc(k, sp_energies, scissor);
        arma::cx_mat Ak_mat = ResultGTF.Aexc.rows(ResultGTF.dimkblock*kind, ResultGTF.dimkblock*(kind+1) - 1);

        arma::cx_mat vk_vectorised = arma::repmat(vk.as_col(), 1, nA);
        arma::cx_mat Vk_exc_vec = arma::conj( arma::repmat(Ak_mat, 3, 1) ) % vk_vectorised;    
        Vk_exc_X += arma::sum(Vk_exc_vec.rows(0, ResultGTF.dimkblock - 1));                         //sum over v,c; for component X
        Vk_exc_Y += arma::sum(Vk_exc_vec.rows(ResultGTF.dimkblock, 2*ResultGTF.dimkblock - 1));     //sum over v,c; for component Y
        Vk_exc_Z += arma::sum(Vk_exc_vec.rows(2*ResultGTF.dimkblock, 3*ResultGTF.dimkblock - 1));   //sum over v,c; for component Z
    }
    // Reduce the k-summation 
    std::vector<double> Vk_exc_proc_X = arma::conv_to<std::vector<double>>::from( arma::join_horiz(arma::real(Vk_exc_X), arma::imag(Vk_exc_X)) );
    std::vector<double> Vk_exc_proc_Y = arma::conv_to<std::vector<double>>::from( arma::join_horiz(arma::real(Vk_exc_Y), arma::imag(Vk_exc_Y)) );
    std::vector<double> Vk_exc_proc_Z = arma::conv_to<std::vector<double>>::from( arma::join_horiz(arma::real(Vk_exc_Z), arma::imag(Vk_exc_Z)) );
    std::vector<double> Vk_exc_tot_X(2*nA); 
    std::vector<double> Vk_exc_tot_Y(2*nA); 
    std::vector<double> Vk_exc_tot_Z(2*nA); 
    MPI_Barrier (MPI_COMM_WORLD);
    MPI_Reduce(&Vk_exc_proc_X[0], &Vk_exc_tot_X[0], 2*nA, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Vk_exc_proc_Y[0], &Vk_exc_tot_Y[0], 2*nA, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Vk_exc_proc_Z[0], &Vk_exc_tot_Z[0], 2*nA, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(procMPI_rank == 0){
        arma::cx_rowvec Vk_exc_totArma_X, Vk_exc_totArma_Y, Vk_exc_totArma_Z;
        {
            arma::rowvec Vk_exc_tot_X_pre = arma::conv_to<arma::rowvec>::from(Vk_exc_tot_X);
            arma::rowvec Vk_exc_tot_Y_pre = arma::conv_to<arma::rowvec>::from(Vk_exc_tot_Y);
            arma::rowvec Vk_exc_tot_Z_pre = arma::conv_to<arma::rowvec>::from(Vk_exc_tot_Z);
            Vk_exc_totArma_X = Vk_exc_tot_X_pre.subvec(0,nA-1) + ResultGTF.imag*Vk_exc_tot_X_pre.subvec(nA,2*nA-1); //eq (42) in original XATU paper, for X component and with exciton states by columns
            Vk_exc_totArma_Y = Vk_exc_tot_Y_pre.subvec(0,nA-1) + ResultGTF.imag*Vk_exc_tot_Y_pre.subvec(nA,2*nA-1); //eq (42) in original XATU paper, for Y component and with exciton states by columns
            Vk_exc_totArma_Z = Vk_exc_tot_Z_pre.subvec(0,nA-1) + ResultGTF.imag*Vk_exc_tot_Z_pre.subvec(nA,2*nA-1); //eq (42) in original XATU paper, for Z component and with exciton states by columns
        }
        arma::cx_mat Vk_exc_totArma_X_Edelta = arma::repmat(Vk_exc_totArma_X % Eexc_inv, nomega, 1) % delta_meshgrid;
        arma::cx_mat Vk_exc_totArma_Y_Edelta = arma::repmat(Vk_exc_totArma_Y % Eexc_inv, nomega, 1) % delta_meshgrid;
        arma::cx_mat Vk_exc_totArma_Z_Edelta = arma::repmat(Vk_exc_totArma_Z % Eexc_inv, nomega, 1) % delta_meshgrid;

        arma::mat sigma_diag = arma::zeros<arma::mat>(nomega, ResultGTF.ndim); //store the (real) diagonal sigma components resolved in omega, concatenated as (XX,YY,ZZ) until ndim
        sigma_diag.col(0) += arma::real( arma::sum( Vk_exc_totArma_X_Edelta % arma::repmat(arma::conj(Vk_exc_totArma_X), nomega,1), 1) );
        if(ResultGTF.ndim >= 2){
            sigma_diag.col(1) += arma::real( arma::sum( Vk_exc_totArma_Y_Edelta % arma::repmat(arma::conj(Vk_exc_totArma_Y), nomega,1), 1) );
            if(ResultGTF.ndim == 3){
                sigma_diag.col(2) += arma::real( arma::sum( Vk_exc_totArma_Z_Edelta % arma::repmat(arma::conj(Vk_exc_totArma_Z), nomega,1), 1) );
            }
        }
        sigma_diag *= spin_degeneracy_fac/(ResultGTF.nkBSE * ResultGTF.unitCellVolume*std::pow(ANG2AU,ResultGTF.ndim));
    
        // Write absorption components to file
        // (first clear the corresponding file if already present, then write the header and finally the calculated values)
        savefile = "Results/4-Absorption/" + savefile + ".absor";
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
        output_file << "ABSORPTION DIAGONAL COMPONENTS (DELTA FUNCTION)" << std::endl;
        output_file << "nk = " << ResultGTF.nkBSE << ", nkPol = " << ResultGTF.nkPol << std::endl;
        output_file << "Valence bands: " << ResultGTF.valenceBands.t();
        output_file << "Conduction bands: " << ResultGTF.conductionBands.t();
        output_file << "Broadening (eV): " << broadening/EV2HARTREE << ", Units: " << units_str << std::endl;
        output_file << "Frequency (eV)" + comp_dim_str << std::endl;
        output_file << "Number of excitons included: " << nA << std::endl;
        output_file << "Number of frequency points: " << std::endl;
        output_file << nomega << std::endl;
        output_file.precision(12);
        output_file << std::scientific;
        for(uint32_t omega_ind = 0; omega_ind < nomega; omega_ind++){
            if(ResultGTF.ndim == 1){
                output_file << omegaDom(omega_ind) << "   " << sigma_diag(omega_ind, 0) << std::endl;
            }
            else if(ResultGTF.ndim == 2){
                output_file << omegaDom(omega_ind) << "   " << sigma_diag(omega_ind, 0) << " " << sigma_diag(omega_ind, 1) << std::endl;
            }
            else{
                output_file << omegaDom(omega_ind) << "   " << sigma_diag(omega_ind, 0) << " " << 
                    sigma_diag(omega_ind, 1) << " " << sigma_diag(omega_ind, 2) << std::endl;
            }
        }
        output_file.close();
        std::cout << "Done! Diagonal absorption components stored in " << savefile << std::endl;
    }
    MPI_Barrier (MPI_COMM_WORLD);


    MPI_Finalize();
    return 0;

}
