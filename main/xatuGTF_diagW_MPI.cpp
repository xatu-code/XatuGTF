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
        std::cout << "|                           BSE k=k' TERMS (DIRECT)                         |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp";
    int ncells = 43;                                  //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string exciton_file = "InputFiles/Pho_88x66_alpha.txt";
    uint metric = 1;                                  //0 for the overlap metric, 1 for the attenuated Coulomb metric (integrals must be pre-computed)
    std::string intName = "custom";    //For the metric (2C & 3C) and Ewald/Coulomb (2C)
    std::string savefile = "custom_100x100";  //Result will be saved in file Results/2-BSEHamiltonian/savefile.diag
    bool saveGridFrac = true;                         //Store the BSE k-grid (in fractional coordinates) or not, in Results/2-BSEHamiltonian/BSE_frac.grid
    bool saveGridAng = true;                          //Store the BSE k-grid (in Angstrom) or not, in Results/2-BSEHamiltonian/BSE_Ang.grid
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
   
    uint procs_per_Deltak   = procMPI_size / nDeltak;  // number of MPI processes attributed to each Deltak displacement 
    uint Deltak_thisproc    = procMPI_rank % nDeltak;  // index of the Deltak displacement attributed to this MPI process
    uint kinterval_thisproc = procMPI_rank / nDeltak;  // index of the chunks of k-points assigned to this process for the corresponding Deltak
    uint32_t nk_per_proc    = GTFexc.nkBSE / procs_per_Deltak; // minimum number of k-points in BSE grid assigned to each processor (w/o remainders) 
    uint nk_remainder       = GTFexc.nkBSE % procs_per_Deltak; // number of remainding k-points after first assignment; to be distributed starting from the first MPI process
    uint p1 = std::min(nk_remainder, kinterval_thisproc);   // number of previous k-intervals containing one more remainding k-point 
    uint p2 = std::min(nk_remainder, kinterval_thisproc + 1);   
    
    if(nk_per_proc == 0){
        std::cout << "There are idle MPI processes. Reduce the number to no more than " << nDeltak*GTFexc.nkBSE << std::endl;
        throw std::invalid_argument("");
    }
    arma::mat kpoints_thisproc;
    if(procMPI_size % nDeltak == 0){
        kpoints_thisproc = GTFexc.kpointsBSE.cols( nk_per_proc*kinterval_thisproc + p1, nk_per_proc*(kinterval_thisproc + 1) - 1 + p2 );
    }
    else{
        std::cout << "The number of MPI processes must be a multiple of " << nDeltak << std::endl;
        throw std::invalid_argument("");
    }
    uint32_t nk_thisproc = kpoints_thisproc.n_cols; 
    uint32_t length_realDk = nk_thisproc*GTFexc.dimkblock_triang;
    std::vector<double> Dk_Deltak(2*length_realDk);

    // Compute the part of the assigned Deltak displacement
    arma::colvec Deltak = Deltakpoints.col(Deltak_thisproc);
    arma::cx_mat JMprod = GTFexc.computeJMproduct(Deltak);       // J^{1/2}(-(k-k')) * M^{-1}(-(k-k')) matrix in AUX basis
    arma::cx_mat Pik    = GTFexc.computePik(Deltak, JMprod);      // Pi(k-k') matrix (auxiliary static irreducible polarizability) in AUX basis
    arma::cx_mat inv_dielectric_mat;                                   // Inverse of the static dielectric matrix in AUX basis at k-k'
    bool inv_em_bool = arma::inv_sympd(inv_dielectric_mat, arma::eye(GTFexc.norbitals_AUX,GTFexc.norbitals_AUX) - Pik);
    if(!inv_em_bool){ 
        std::cerr << "WARNING! Dielectric matrix not positive definite" << std::endl;
        inv_em_bool = arma::inv(inv_dielectric_mat, arma::eye(GTFexc.norbitals_AUX,GTFexc.norbitals_AUX) - Pik); 
        if(!inv_em_bool){
            throw std::logic_error("ERROR: Static dielectric matrix in the AUX basis appears to be singular");
        }
    }
        
    double prefac = 1./(nDeltak * GTFexc.nkBSE);
    for(uint32_t kind = 0; kind < nk_thisproc; kind++){ // iterate over k-points 
        arma::colvec k = kpoints_thisproc.col(kind);
        arma::colvec kD = k + Deltak;
        uint32_t kind_pastblocks = kind*GTFexc.dimkblock_triang;

        arma::colvec eigval1;
        arma::cx_mat eigvec1; 
        GTFexc.solveBands(k,  eigval1, eigvec1);
        arma::cx_mat eigvec1v = eigvec1.cols(GTFexc.valenceBands);
        arma::cx_mat eigvec1c = eigvec1.cols(GTFexc.conductionBands);

        arma::cx_mat eiKR2mat = arma::zeros<arma::cx_mat>(GTFexc.ncellsM3c + 1,GTFexc.ncellsM3c + 1);
        for(uint i1 = 0; i1 <= GTFexc.ncellsM3c; i1++){
            double k1R1 = arma::dot(k, GTFexc.RlistFrac_M3c.col(i1));
            for(uint i2 = 0; i2 <= GTFexc.ncellsM3c; i2++){
                eiKR2mat(i1,i2) = std::exp( GTFexc.imag*TWOPI*(arma::dot(kD, GTFexc.RlistFrac_M3c.col(i2)) - k1R1) );
            }
        }
        arma::cx_mat vPmuk1nuk2_pre = arma::zeros<arma::cx_mat>(GTFexc.norbitals_AUX, GTFexc.norbitals*GTFexc.norbitals);
        for(uint64_t i = 0; i < GTFexc.nvaluesM3c; i++){
            vPmuk1nuk2_pre(GTFexc.metric3CIndices[i][0], GTFexc.metric3CIndices[i][1] + GTFexc.norbitals*GTFexc.metric3CIndices[i][2]) += GTFexc.metric3CValues[i] * eiKR2mat(GTFexc.metric3CIndices[i][3], GTFexc.metric3CIndices[i][4]);
        }
        vPmuk1nuk2_pre = JMprod * vPmuk1nuk2_pre;
        arma::cx_mat vPvv = ( vPmuk1nuk2_pre * arma::kron(eigvec1v, arma::conj(eigvec1v)) ).st();
        arma::cx_mat vPcc = vPmuk1nuk2_pre * arma::kron(eigvec1c, arma::conj(eigvec1c));
            
        for(uint s = 0; s < GTFexc.dimkblock_triang; s++){
            uint r = GTFexc.dimkblock_triang - (s+1);
            uint l = (std::sqrt(8*r + 1) - 1)/2;
            uint i = (s + GTFexc.dimkblock + (l*(l+1))/2 ) - GTFexc.dimkblock_triang;
            uint j = GTFexc.dimkblock - (l+1);
            uint v0ind = i % GTFexc.nvbands;  
            uint c0ind = i / GTFexc.nvbands; 
            uint v1ind = j % GTFexc.nvbands;  
            uint c1ind = j / GTFexc.nvbands; 
            double Efac = (i == j)? (eigval1(GTFexc.conductionBands(0) + c0ind) + GTFexc.scissor - eigval1(GTFexc.valenceBands(0) + v0ind))/nDeltak : 0.;
            std::complex<double> Dij1 = arma::cdot( inv_dielectric_mat*(vPcc.col(GTFexc.ncbands*c1ind + c0ind)), vPvv.row(GTFexc.nvbands*v1ind + v0ind) );
            std::complex<double> Dij2 = arma::cdot( vPvv.row(GTFexc.nvbands*v0ind + v1ind), inv_dielectric_mat*vPcc.col(GTFexc.ncbands*c0ind + c1ind) );
            Dk_Deltak[kind_pastblocks + s]                 = Efac - std::real(Dij1 + Dij2)*prefac;
            Dk_Deltak[kind_pastblocks + s + length_realDk] = - std::imag(Dij1 + Dij2)*prefac;
        }
    }
    MPI_Barrier (MPI_COMM_WORLD);

    // Reduce the results, averaging over the Deltak. First, each segment of k-points is grouped with the same segment of the other Deltak
    MPI_Comm MPI_NEWCOMM;
    MPI_Comm_split (MPI_COMM_WORLD, kinterval_thisproc, procMPI_rank, &MPI_NEWCOMM);
    std::vector<double> Dk_Delta_vec_total(2*length_realDk);
    MPI_Reduce(&Dk_Deltak[0], &Dk_Delta_vec_total[0], 2*length_realDk, MPI_DOUBLE, MPI_SUM, 0, MPI_NEWCOMM);
    MPI_Barrier (MPI_COMM_WORLD);

    // Write direct term diagonal entries to file
    savefile = "Results/2-BSEHamiltonian/" + savefile + ".diag";
    if(procMPI_rank == 0) { // first clear the corresponding file if already present, then write the header
        std::ofstream output_file0(savefile, std::ios::trunc);
        output_file0.close();
        std::ofstream output_file(savefile, std::ios::app);
        output_file << "DIAGONAL W BLOCKS LOWER TRIANGLE" << std::endl;
        output_file << "nk = " << GTFexc.nkBSE << ", nkPol = " << GTFexc.nkPol << ", Integrals name: " << intName << ", Deltak = " << static_cast<uint>(1/DK) << std::endl;
        output_file << "Valence bands: " << GTFexc.valenceBands.t();
        output_file << "Conduction bands: " << GTFexc.conductionBands.t();
        output_file << "Entry (Re), Entry (Im), Row, Column" << std::endl;
        output_file << GTFexc.dimBSE << std::endl;
        output_file << GTFexc.nkBSE*GTFexc.dimkblock_triang << std::endl;
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
    for(uint k_chunk = 0; k_chunk < procs_per_Deltak; k_chunk++){
        if(static_cast<uint>(procMPI_rank) == nDeltak*k_chunk){
            uint32_t past_kchunks = (nk_per_proc*k_chunk + p1)*GTFexc.dimkblock;
            std::ofstream output_file(savefile, std::ios::app);
            output_file.precision(12);
            output_file << std::scientific;
            for(uint32_t ent = 0; ent < length_realDk; ent++){
                uint ent_pastblocks = ent / GTFexc.dimkblock_triang;
                uint n = ent % GTFexc.dimkblock_triang;     
                uint r = GTFexc.dimkblock_triang - (n + 1);
                uint l = (std::sqrt(8*r + 1) - 1)/2;
                output_file << Dk_Delta_vec_total[ent] << " " << Dk_Delta_vec_total[ent + length_realDk] << "  " << 
                    past_kchunks + ((ent_pastblocks + 1)*GTFexc.dimkblock + n + (l*(l+1))/2) - GTFexc.dimkblock_triang << " " << 
                        past_kchunks + (ent_pastblocks + 1)*GTFexc.dimkblock - (l+1) << std::endl;
            }
            output_file.close();
        }
        MPI_Barrier (MPI_COMM_WORLD);
    }

    if(saveGridFrac && procMPI_rank == 0){
        arma::mat BSEgrid_red = GTFexc.kpointsBSE.rows(0, GTFexc.ndim - 1);
        BSEgrid_red.save("Results/2-BSEHamiltonian/BSE_frac.grid", arma::arma_ascii);
    }
    if(saveGridAng && procMPI_rank == 0){
        arma::mat BSEgrid_Ang = (GTFexc.Gbasis*( GTFexc.kpointsBSE.rows(0, GTFexc.ndim - 1) ));
        BSEgrid_Ang = BSEgrid_Ang.rows(0, GTFexc.ndim - 1);
        BSEgrid_Ang.save("Results/2-BSEHamiltonian/BSE_Ang.grid", arma::arma_ascii);
    }

    MPI_Finalize();
    return 0;

}
