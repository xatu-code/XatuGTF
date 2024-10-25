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
        std::cout << "|                              BSE k/=k' TERMS                              |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/hBN_HSE06.outp";
    int ncells = 179;                                 //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string exciton_file = "InputFiles/hBN_GAUSSIAN.txt";
    uint metric = 0;                                  //0 for the overlap metric, 1 for the attenuated Coulomb metric (integrals must be pre-computed)
    std::string intName = "NEW_def2-TZVPPD-RIFIT";    //For the metric (2C & 3C) and Ewald/Coulomb (2C)
    std::string savefile = "BSE";                     //Result will be saved in file Results/2-BSEHamiltonian/savefile.offdiag
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationExciton_MPI Excitonconfig(exciton_file, procMPI_rank, procMPI_size);
    xatu::ExcitonGTF_MPI GTFexc(Excitonconfig, CRYSTALconfig, procMPI_rank, procMPI_size, metric, false, intName);

    // Distribute k-chunks among MPI processes
    uint32_t nkchunks_without_0 = GTFexc.nAbsk - 1;                  // total number of (equal |k-k'|)-sets to be distributed, i.e. excluding the k=k' diagonal
    uint32_t nkchunks_per_proc = nkchunks_without_0 / procMPI_size;  // number of (equal |k-k'|)-sets first distributed to each MPI process, excluding |k-k'|=0   
    int32_t nkchunks_remainder = nkchunks_without_0 % procMPI_size;  // remaining (equal |k-k'|)-sets after first distribution, excluding |k-k'|=0  
    
    arma::ucolvec selected_chunks_thisproc; // indices of the chunks (i.e. |k-k'| sets) corresponding to this MPI process, as given by SystemGTF::generatekentries
    if(nkchunks_per_proc == 0){
        std::cout << "There are idle MPI processes. Reduce the number to no more than " << nkchunks_without_0 << std::endl;
        throw std::invalid_argument("");
    } else { // equal distribution of chunks among processes, excluding |k-k'|=0  
        selected_chunks_thisproc = arma::regspace<arma::ucolvec>(static_cast<uint>(procMPI_rank)*nkchunks_per_proc + 1, static_cast<uint>((procMPI_rank+1))*nkchunks_per_proc);
    }

    uint32_t optimalMPI = std::ceil((double)nkchunks_without_0/(double)(nkchunks_per_proc + 1));
    if( (procMPI_rank == 0) && (nkchunks_remainder != 0) && (optimalMPI != static_cast<uint32_t>(procMPI_size)) ){
        std::cout << "ATTENTION: The performance would be the same with any number of MPI processes in the (closed) iterval [" <<
            optimalMPI << ", " << nkchunks_without_0/nkchunks_per_proc << "]" << std::endl;

    }
    for(int r = 0; r < nkchunks_remainder; r++){ // distribute the remainding chunks, one for each process until depleted
        if(procMPI_rank == r){
            arma::ucolvec selected_chunks_thisproc_remainder = {static_cast<uint>(procMPI_size*nkchunks_per_proc + 1 + r)};
            selected_chunks_thisproc = arma::join_vert(selected_chunks_thisproc, selected_chunks_thisproc_remainder);
        }
    }
    std::vector<std::vector<std::array<uint32_t,3>>> kentries_chunks_thisproc = GTFexc.generatekentries(Excitonconfig.excitonInfoGTF.nki, selected_chunks_thisproc);
    uint32_t nkchunks_thisproc = kentries_chunks_thisproc.size(); // number of chunks finally assigned to this MPI process

    uint dimkblock_square = GTFexc.dimkblock * GTFexc.dimkblock;  // number of entries in each (k,k')-block
    // uint64_t nk_thisproc = 0;                                     // number of k-points (among all chunks) finally assigned to this MPI process
    std::vector<uint32_t> chunk_sizes(nkchunks_thisproc);         // vector with the number of H entries for each |k-k'|-chunk
    for(uint32_t chunk = 0; chunk < nkchunks_thisproc; chunk++){
        chunk_sizes[chunk] = dimkblock_square * kentries_chunks_thisproc[chunk].size();
    //    nk_thisproc += kentries_chunks_thisproc[chunk].size();
    }

    if(procMPI_rank == 0){
        std::cout << "Chunks of (k,k') points distributed" << std::endl;
        std::cout << "Total number of chunks (w/o diagonal): " << nkchunks_without_0 << ", chunks per process (max): " << nkchunks_thisproc << " "  << std::endl;
    }

    // Start calculation
    std::vector<std::vector<double>> Hk_vec(nkchunks_thisproc); // vector to store all the H entries (real & imag parts) computed by this process, later written into file
    for(uint32_t chunk = 0; chunk < nkchunks_thisproc; chunk++){
        Hk_vec[chunk].resize(2*chunk_sizes[chunk]);
    }

    for(uint32_t chunk = 0; chunk < nkchunks_thisproc; chunk++){ // iterate over chunks (sets in bijection with all the different |k-k'|)
        // Pre-compute the common quantities to all (k,k') within the present chunk (i.e. with the same |k-k'|)
        arma::colvec kdiff0  = GTFexc.kpointsBSE.col(kentries_chunks_thisproc[chunk][0][0]) - GTFexc.kpointsBSE.col(kentries_chunks_thisproc[chunk][0][1]); //k-k' in fractional coordinates
        bool k0_plus_minus = kentries_chunks_thisproc[chunk][0][2];
        arma::cx_mat JMprod = GTFexc.computeJMproduct(kdiff0);            // J^{1/2}(k-k') * M^{-1}(k-k') matrix in AUX basis
        arma::cx_mat JMprod_conj = arma::conj(JMprod);                    // J^{1/2}(-(k-k')) * M^{-1}(-(k-k')) matrix in AUX basis
        arma::cx_mat Pik    = GTFexc.computePik(kdiff0, JMprod);     // Pi(k-k') matrix (auxiliary static irreducible polarizability) in AUX basis
        arma::cx_mat inv_dielectric_mat;                                  // Inverse of the static dielectric matrix in AUX basis at k-k'
        bool inv_em_bool = arma::inv_sympd(inv_dielectric_mat, arma::eye(GTFexc.norbitals_AUX,GTFexc.norbitals_AUX) - Pik);
        if(!inv_em_bool){ 
            std::cerr << "WARNING! Dielectric matrix not positive definite" << std::endl;
            inv_em_bool = arma::inv(inv_dielectric_mat, arma::eye(GTFexc.norbitals_AUX,GTFexc.norbitals_AUX) - Pik); 
            if(!inv_em_bool){
                throw std::logic_error("ERROR: Static dielectric matrix in the AUX basis appears to be singular");
            }
        }
        
        for(uint32_t kpair = 0; kpair < kentries_chunks_thisproc[chunk].size(); kpair++){ // iterate over all the (k,k') pairs within the present chunk (i.e. with the same |k-k'|)       
            uint32_t kind_pastblocks = kpair*dimkblock_square;
            arma::colvec k1 = GTFexc.kpointsBSE.col(kentries_chunks_thisproc[chunk][kpair][0]);  // k in fractional coordinates
            arma::colvec k2 = GTFexc.kpointsBSE.col(kentries_chunks_thisproc[chunk][kpair][1]);  // k' in fractional coordinates
            bool k_plus_minus = kentries_chunks_thisproc[chunk][kpair][2];

            arma::colvec eigval1, eigval2;
            arma::cx_mat eigvec1, eigvec2; 
            GTFexc.solveBands(k1, eigval1, eigvec1);
            GTFexc.solveBands(k2, eigval2, eigvec2);
            arma::cx_mat eigvec1v = eigvec1.cols(GTFexc.valenceBands);
            arma::cx_mat eigvec2v = eigvec2.cols(GTFexc.valenceBands);
            arma::cx_mat eigvec1c = eigvec1.cols(GTFexc.conductionBands);
            arma::cx_mat eigvec2c = eigvec2.cols(GTFexc.conductionBands);

            arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(GTFexc.ncellsM3c + 1,GTFexc.ncellsM3c + 1);
            for(uint i1 = 0; i1 <= GTFexc.ncellsM3c; i1++){
                double k1R1 = arma::dot(k1, GTFexc.RlistFrac_M3c.col(i1));
                for(uint i2 = 0; i2 <= GTFexc.ncellsM3c; i2++){
                    eiKR2mat(i1,i2) = std::exp( GTFexc.imag*TWOPI*(arma::dot(k2, GTFexc.RlistFrac_M3c.col(i2)) - k1R1) );
                }
            }

            arma::cx_mat vPmuk1nuk2_pre = arma::zeros<arma::cx_mat>(GTFexc.norbitals_AUX, GTFexc.norbitals*GTFexc.norbitals);
            for(uint64_t i = 0; i < GTFexc.nvaluesM3c; i++){
                vPmuk1nuk2_pre(GTFexc.metric3CIndices[i][0], GTFexc.metric3CIndices[i][1] + GTFexc.norbitals*GTFexc.metric3CIndices[i][2]) += GTFexc.metric3CValues[i] * eiKR2mat(GTFexc.metric3CIndices[i][3], GTFexc.metric3CIndices[i][4]);
            }

            if(k_plus_minus == k0_plus_minus){  // k-k' = k_0 - k'_0
                vPmuk1nuk2_pre = JMprod_conj * vPmuk1nuk2_pre;
                arma::cx_mat vPvv = ( vPmuk1nuk2_pre * arma::kron(eigvec2v, arma::conj(eigvec1v)) ).st();
                arma::cx_mat vPcc = vPmuk1nuk2_pre * arma::kron(eigvec2c, arma::conj(eigvec1c));
                for(uint s = 0; s < dimkblock_square; s++){
                    uint i = s % GTFexc.dimkblock;
                    uint j = s / GTFexc.dimkblock;
                    uint v0ind = i % GTFexc.nvbands;  
                    uint c0ind = i / GTFexc.nvbands; 
                    uint v1ind = j % GTFexc.nvbands;  
                    uint c1ind = j / GTFexc.nvbands; 
                    std::complex<double> Dij = arma::cdot( vPcc.col(GTFexc.ncbands*c1ind + c0ind), vPvv.row(GTFexc.nvbands*v1ind + v0ind)*inv_dielectric_mat );
                    Hk_vec[chunk][kind_pastblocks + s]                      = - std::real(Dij)/GTFexc.nkBSE;
                    Hk_vec[chunk][kind_pastblocks + s + chunk_sizes[chunk]] = - std::imag(Dij)/GTFexc.nkBSE;
                }
            } 
            else {  // k-k' = - (k_0 - k'_0)
                vPmuk1nuk2_pre = JMprod * vPmuk1nuk2_pre;
                arma::cx_mat vPvv = ( vPmuk1nuk2_pre * arma::kron(eigvec2v, arma::conj(eigvec1v)) ).st();
                arma::cx_mat vPcc = vPmuk1nuk2_pre * arma::kron(eigvec2c, arma::conj(eigvec1c));
                for(uint s = 0; s < dimkblock_square; s++){
                    uint i = s % GTFexc.dimkblock;
                    uint j = s / GTFexc.dimkblock;
                    uint v0ind = i % GTFexc.nvbands;  
                    uint c0ind = i / GTFexc.nvbands; 
                    uint v1ind = j % GTFexc.nvbands;  
                    uint c1ind = j / GTFexc.nvbands; 
                    std::complex<double> Dij = arma::cdot( inv_dielectric_mat*vPcc.col(GTFexc.ncbands*c1ind + c0ind), vPvv.row(GTFexc.nvbands*v1ind + v0ind) );
                    Hk_vec[chunk][kind_pastblocks + s]                      = - std::real(Dij)/GTFexc.nkBSE;
                    Hk_vec[chunk][kind_pastblocks + s + chunk_sizes[chunk]] = - std::imag(Dij)/GTFexc.nkBSE;
                }
            }

        }

    }
    MPI_Barrier (MPI_COMM_WORLD);

    // Write non-diagonal Hamiltonian entries to file
    savefile = "Results/2-BSEHamiltonian/" + savefile + ".offdiag";
    if(procMPI_rank == 0) { // first clear the corresponding file if already present, then write the header
        std::ofstream output_file0(savefile, std::ios::trunc);
        output_file0.close();
        std::ofstream output_file(savefile, std::ios::app);
        output_file << "NON-DIAGONAL W BLOCKS" << std::endl;
        output_file << "nk = " << GTFexc.nkBSE << ", nkPol = " << GTFexc.nkPol << ", Integrals name: " << intName << std::endl;
        output_file << "Valence bands: " << GTFexc.valenceBands.t();
        output_file << "Conduction bands: " << GTFexc.conductionBands.t();
        output_file << "Entry (Re), Entry (Im), Row, Column" << std::endl;
        output_file << GTFexc.dimBSE << std::endl;
        output_file << ((GTFexc.nkBSE*(GTFexc.nkBSE - 1))/2)*dimkblock_square << std::endl;
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    for(int r = 0; r < procMPI_size; r++){
        if(procMPI_rank == r){
            std::ofstream output_file(savefile, std::ios::app);
            output_file.precision(12);
            output_file << std::scientific;
            for(uint32_t chunk = 0; chunk < nkchunks_thisproc; chunk++){
                for(uint32_t ent = 0; ent < chunk_sizes[chunk]; ent++){
                    uint32_t kpair = ent / dimkblock_square;
                    uint32_t k1ind = kentries_chunks_thisproc[chunk][kpair][0];
                    uint32_t k2ind = kentries_chunks_thisproc[chunk][kpair][1];
                    uint n = ent % dimkblock_square;
                    output_file << Hk_vec[chunk][ent] << " " << Hk_vec[chunk][ent + chunk_sizes[chunk]] << "  " <<
                        k1ind*GTFexc.dimkblock + (n % GTFexc.dimkblock) << " " << k2ind*GTFexc.dimkblock + (n / GTFexc.dimkblock) << std::endl;
                }
            }
            output_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }


    MPI_Finalize();
    return 0;

}
