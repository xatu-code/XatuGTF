// #define ARMA_ALLOW_FAKE_GCC
#include <mpi.h>
#include "xatu.hpp"

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
        std::cout << "|                         BSE k=k' TERMS (EXCHANGE)                         |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp"; 
    int ncells = 43;                                 //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string exciton_file = "InputFiles/Pho_88x66_alpha.txt";
    uint metric = 0;                                  //0 for the overlap metric, 1 for the attenuated Coulomb metric (integrals must be pre-computed)
    std::string intName = "custom";    //For the metric (2C & 3C) and Ewald/Coulomb (2C)
    std::string savefile = "custom_100x100";                     //Result will be saved in file Results/2-BSEHamiltonian/savefile.exch
    // Deltak points 
    double DK = 1./100.;                                  //Deltak in fractional coordinates
    arma::mat Deltakpoints = {{DK,0.},{0.,DK}};  //In frac coordinates, not containing opposites
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationExciton_MPI Excitonconfig(exciton_file, procMPI_rank, procMPI_size);
    xatu::ExcitonGTF_MPI GTFexc(Excitonconfig, CRYSTALconfig, procMPI_rank, procMPI_size, metric, true, intName);

    Deltakpoints = Deltakpoints.t(); //list now by columns
    Deltakpoints = arma::join_horiz(Deltakpoints, -Deltakpoints); //add opposite points to the list
    uint nDeltak = Deltakpoints.n_cols;

    uint64_t nk_triang = (GTFexc.nkBSE*(GTFexc.nkBSE + 1))/2;
    uint dimkblock_square = GTFexc.dimkblock * GTFexc.dimkblock;  // number of entries in each (k,k')-block
   
    uint procs_per_Deltak      = procMPI_size / nDeltak;  // number of MPI processes attributed to each Deltak displacement 
    uint Deltak_thisproc       = procMPI_rank % nDeltak;  // index of the Deltak displacement attributed to this MPI process
    uint interval_blocks_thisproc = procMPI_rank / nDeltak;  // index of the subset of (k,k')-blocks assigned to this process for the corresponding Deltak
    uint32_t blocks_per_proc   = nk_triang / procs_per_Deltak; // minimum number of (k,k')-blocks assigned to each processor (w/o remainders) 
    uint blocks_remainder      = nk_triang % procs_per_Deltak; // number of remainding (k,k')-blocks after first assignment; to be distributed starting from the first MPI process
    uint p1 = std::min(blocks_remainder, interval_blocks_thisproc);
    uint p2 = std::min(blocks_remainder, interval_blocks_thisproc + 1);
    
    if(procMPI_size % nDeltak != 0){
        std::cout << "The number of MPI processes must be a multiple of " << nDeltak << std::endl;
        throw std::invalid_argument("");
    }
    if(blocks_per_proc == 0){
        std::cout << "There are idle MPI processes. Reduce the number to no more than " << nDeltak*nk_triang << std::endl;
        throw std::invalid_argument("");
    }
    
    uint32_t blocks_thisproc = (interval_blocks_thisproc < blocks_remainder)? (blocks_per_proc + 1) : blocks_per_proc;
    uint32_t length_realXk = blocks_thisproc * dimkblock_square;
    std::vector<double> Xk_Deltak(2*length_realXk);

    // Compute the part of the assigned Deltak displacement
    arma::colvec Deltak = Deltakpoints.col(Deltak_thisproc);
    arma::cx_mat JMprod = GTFexc.computeJMproduct(Deltak);       // J^{1/2}(-(k-k')) * M^{-1}(-(k-k')) matrix in AUX basis

    arma::cx_mat vPvc_k_mat = arma::zeros<arma::cx_mat>(GTFexc.norbitals_AUX, GTFexc.nkBSE * GTFexc.dimkblock); // matrix with all the v_{R}^{vk,ck+Deltak} for the corresponding Deltak, with different k's concatenated horizontally
    for(uint32_t kind = 0; kind < GTFexc.nkBSE; kind++){ // iterate over k-points 
        arma::colvec k = GTFexc.kpointsBSE.col(kind);
        arma::colvec kD = k + Deltak;

        arma::colvec eigval1;
        arma::cx_mat eigvec1; 
        GTFexc.solveBands(k, eigval1, eigvec1);
        arma::cx_mat eigvec1v = eigvec1.cols(GTFexc.valenceBands);
        arma::cx_mat eigvec1c = eigvec1.cols(GTFexc.conductionBands);

        arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(GTFexc.ncellsM3c + 1,GTFexc.ncellsM3c + 1);
        for(uint i1 = 0; i1 <= GTFexc.ncellsM3c; i1++){
            double k1R1 = arma::dot(k, GTFexc.RlistFrac_M3c.col(i1));
            for(uint i2 = 0; i2 <= GTFexc.ncellsM3c; i2++){
                eiKR2mat(i1,i2) = std::exp( GTFexc.imag*TWOPI*(arma::dot(kD, GTFexc.RlistFrac_M3c.col(i2)) - k1R1) );
            }
        }
        arma::cx_mat vPmunu_pre_k = arma::zeros<arma::cx_mat>(GTFexc.norbitals_AUX, GTFexc.norbitals*GTFexc.norbitals);
        for(uint64_t i = 0; i < GTFexc.nvaluesM3c; i++){
            vPmunu_pre_k(GTFexc.metric3CIndices[i][0], GTFexc.metric3CIndices[i][1] + GTFexc.norbitals*GTFexc.metric3CIndices[i][2]) += GTFexc.metric3CValues[i] * eiKR2mat(GTFexc.metric3CIndices[i][3], GTFexc.metric3CIndices[i][4]);
        }
        vPmunu_pre_k = JMprod * vPmunu_pre_k;
        vPvc_k_mat.cols(kind*GTFexc.dimkblock, (kind + 1)*GTFexc.dimkblock - 1) = vPmunu_pre_k * arma::kron(eigvec1c, arma::conj(eigvec1v));
    }
        
    double prefac = 1./(nDeltak * GTFexc.nkBSE);
    uint32_t previous_blocks = 0;
    for(uint32_t block_ind = interval_blocks_thisproc*blocks_per_proc + p1; block_ind < (interval_blocks_thisproc+1)*blocks_per_proc + p2; block_ind++){ // iterate over (k,k')-blocks 
        uint32_t r = nk_triang - (block_ind + 1);
        uint32_t l = (std::sqrt(8*r + 1) - 1)/2;
        uint32_t k1ind = (block_ind + GTFexc.nkBSE + (l*(l+1))/2 ) - nk_triang;
        uint32_t k2ind = GTFexc.nkBSE - (l + 1);

        arma::cx_mat vPvc_k1 = ( vPvc_k_mat.cols(k1ind*GTFexc.dimkblock, (k1ind + 1)*GTFexc.dimkblock - 1) ).st(); // NOT conjugated
        arma::cx_mat vPvc_k2 = vPvc_k_mat.cols(k2ind*GTFexc.dimkblock, (k2ind + 1)*GTFexc.dimkblock - 1);

        arma::cx_mat Xblock = vPvc_k1 * arma::conj(vPvc_k2);
        for(uint s = 0; s < dimkblock_square; s++){
            uint i = s % GTFexc.dimkblock;
            uint j = s / GTFexc.dimkblock;
            Xk_Deltak[previous_blocks + s]                 = std::real(Xblock(i,j))*prefac;
            Xk_Deltak[previous_blocks + s + length_realXk] = std::imag(Xblock(i,j))*prefac;
        }
        
        previous_blocks += dimkblock_square;
    }
    MPI_Barrier (MPI_COMM_WORLD);

    // Reduce the results, averaging over the Deltak. First, each interval of (k,k')-blocks is grouped with the same interval of the other Deltak
    MPI_Comm MPI_NEWCOMM;
    MPI_Comm_split (MPI_COMM_WORLD, interval_blocks_thisproc, procMPI_rank, &MPI_NEWCOMM);
    std::vector<double> Xk_Delta_vec_total(2*length_realXk);
    MPI_Reduce(&Xk_Deltak[0], &Xk_Delta_vec_total[0], 2*length_realXk, MPI_DOUBLE, MPI_SUM, 0, MPI_NEWCOMM);
    MPI_Barrier (MPI_COMM_WORLD);

    // Write exchange Hamiltonian (lower triangle) entries to file
    savefile = "Results/2-BSEHamiltonian/" + savefile + ".exch";
    if(procMPI_rank == 0) { // first clear the corresponding file if already present, then write the header
        std::ofstream output_file0(savefile, std::ios::trunc);
        output_file0.close();
        std::ofstream output_file(savefile, std::ios::app);
        output_file << "X HAMILTONIAN LOWER TRIANGLE" << std::endl;
        output_file << "nk = " << GTFexc.nkBSE << ", Integrals name: " << intName << ", Deltak = " << static_cast<uint>(1/DK) << std::endl;
        output_file << "Valence bands: " << GTFexc.valenceBands.t();
        output_file << "Conduction bands: " << GTFexc.conductionBands.t();
        output_file << "Entry (Re), Entry (Im), Row, Column" << std::endl;
        output_file << GTFexc.dimBSE << std::endl;
        output_file << (GTFexc.dimBSE*(GTFexc.dimBSE + 1))/2 << std::endl;
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    for(uint block_interval = 0; block_interval < procs_per_Deltak; block_interval++){ //iterate over intervals of (k,k')-blocks
        if(static_cast<uint>(procMPI_rank) == nDeltak*block_interval){
            uint32_t blocks_past_intervals = blocks_per_proc*block_interval + p1;  //total number of (k,k')-blocks in the previous intervals 
            std::ofstream output_file(savefile, std::ios::app);
            output_file.precision(12);
            output_file << std::scientific;
            for(uint32_t ent = 0; ent < length_realXk; ent++){ //iterate over (half) the entries in the vector with X elements  
                uint ent_bands = ent % dimkblock_square;                   //index (vectorised, columns first) within the (k,k')-block associated to the current entry
                uint32_t ent_block = ent / dimkblock_square;                       //index of the (k,k')-block associated to the current entry
                uint32_t total_block = blocks_past_intervals + ent_block;          //definitive index of the (k,k')-block considering the previous intervals
                uint32_t r = nk_triang - (total_block + 1);
                uint32_t l = (std::sqrt(8*r + 1) - 1)/2;
                uint32_t k1ind = total_block - nk_triang + GTFexc.nkBSE + (l*(l+1))/2;
                uint32_t k2ind = GTFexc.nkBSE - (l + 1);
                uint32_t i_ind = k1ind*GTFexc.dimkblock + (ent_bands % GTFexc.dimkblock);
                uint32_t j_ind = k2ind*GTFexc.dimkblock + (ent_bands / GTFexc.dimkblock);
                if(i_ind >= j_ind){
                    output_file << Xk_Delta_vec_total[ent] << " " << Xk_Delta_vec_total[ent + length_realXk] << "  " << 
                        i_ind << " " << j_ind << std::endl;
                }
            }
            output_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;

}
