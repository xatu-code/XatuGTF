#include "xatu/IntegralsOverlap2C_MPI.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase object.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed.
 * @param intName Name of the file where the 2-center overlap matrices will be stored as a vector (o2Mat_intName.o2c).
 * @param basis_id True => SCF basis, False => Auxiliary basis.
 */
IntegralsOverlap2C_MPI::IntegralsOverlap2C_MPI(const IntegralsBase& IntBase, const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName, const bool basis_id) : IntegralsBase{IntBase} {

    overlap2Cfun(procMPI_rank, procMPI_size, tol, nR, intName, basis_id);

}

/**
 * Method to compute the overlap matrices in the auxiliary (if basis_id == false) or SCF (if basis_id == true) basis 
 * (<P,0|P',R> or <mu,0|mu',R>) for the first nR Bravais vectors R. These first nR (at least, until the star of vectors is 
 * completed) are generated with Lattice::generateRlist. Each entry above a certain tolerance (10^-tol) is stored in an entry 
 * of a vector (of arrays) along with the corresponding indices: value,mu,mu',R; in that order. The vector is saved in the 
 * o2Mat_intName.o2c file file, and the list of Bravais vectors in fractional coordinates is saved in the RlistFrac_intName.o2c file.
 * Only the lower triangle of each R-matrix is stored; the upper triangle is given by hermiticity in the k-matrix.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed.
 * @param intName Name of the file where the 2-center overlap matrices will be stored as a vector (o2Mat_intName.o2c).
 * @param basis_id True => SCF basis, False => Auxiliary basis.
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsOverlap2C_MPI::overlap2Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR, const std::string& intName, const bool basis_id){

const double PIpow = std::pow(PI,1.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR, combs, "Overlap2C", procMPI_rank);  //convert Bravais vectors from Angstrom to atomic units
uint32_t nR_star = RlistAU.n_cols;

// Discern between basis sets
const uint32_t dimMat {basis_id? dimMat_SCF : dimMat_AUX};
const std::vector<std::vector<int>>& orbitals_info_int {basis_id? orbitals_info_int_SCF : orbitals_info_int_AUX};
const std::vector<std::vector<double>>& orbitals_info_real {basis_id? orbitals_info_real_SCF : orbitals_info_real_AUX};
const std::vector<double>& FAC12 {basis_id? FAC12_SCF : FAC12_AUX};
const std::vector<std::vector<double>>& FAC3 {basis_id? FAC3_SCF : FAC3_AUX};
const std::string basis_string {basis_id? "SCF" : "AUX"};

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat*(dimMat + 1))/2;
uint64_t total_elem = nelem_triang*nR_star;

uint64_t elems_per_proc = total_elem / procMPI_size;
uint elems_remainder = total_elem % procMPI_size;
uint p1 = std::min(elems_remainder, static_cast<uint>(procMPI_rank));
uint p2 = std::min(elems_remainder, static_cast<uint>(procMPI_rank) + 1);

if(procMPI_rank == 0){
    std::cout << "Computing " << nR_star << " " << dimMat << "x" << dimMat << " 2-center overlap matrices in the " << basis_string  << " basis..." << std::flush;
}

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> overlap2Matrices;
    overlap2Matrices.reserve(elems_per_proc + 1);

    for(uint64_t s = procMPI_rank*elems_per_proc + p1; s < (procMPI_rank+1)*elems_per_proc + p2; s++){ //Spans the lower triangle of all the nR_star matrices <P,0|P',R>
        uint64_t sind {s % nelem_triang};    //Index for the corresponding entry in the overlap matrix, irrespective of the specific R
        uint32_t Rind {static_cast<uint32_t>(s / nelem_triang)};         //Position in RlistAU (e.g. 0 for R=0) of the corresponding Bravais vector 
        uint64_t r = nelem_triang - (sind + 1);
        uint64_t l = (std::sqrt(8*r + 1) - 1)/2;
        uint32_t orb_bra = (sind + dimMat + (l*(l+1))/2 ) - nelem_triang;  //Orbital number (<dimMat) of the bra corresponding to the index s 
        uint32_t orb_ket = dimMat - (l + 1);                               //Orbital number (<dimMat) of the ket corresponding to the index s. orb_ket <= orb_bra (lower triangle)
        // arma::colvec R {RlistAU.col(Rind)};  //Bravais vector (a.u.) corresponding to the "s" matrix element
        
        int L_bra  {orbitals_info_int[orb_bra][2]};
        int m_bra  {orbitals_info_int[orb_bra][3]};
        int nG_bra {orbitals_info_int[orb_bra][4]};
        arma::colvec coords_bra {orbitals_info_real[orb_bra][0], orbitals_info_real[orb_bra][1], orbitals_info_real[orb_bra][2]};  //Position (a.u.) of bra atom
        std::vector<int> g_coefs_bra   {g_coefs_.at( L_bra*(L_bra + 1) + m_bra )};

        int L_ket  {orbitals_info_int[orb_ket][2]};
        int m_ket  {orbitals_info_int[orb_ket][3]};
        int nG_ket {orbitals_info_int[orb_ket][4]};
        arma::colvec coords_ket {RlistAU.col(Rind) + arma::colvec{orbitals_info_real[orb_ket][0], orbitals_info_real[orb_ket][1], orbitals_info_real[orb_ket][2]} };  //Position (a.u.) of ket atom
        std::vector<int> g_coefs_ket   {g_coefs_.at( L_ket*(L_ket + 1) + m_ket )};

        double norm_braket {arma::dot(coords_bra - coords_ket, coords_bra - coords_ket)};
        double FAC12_braket = FAC12[orb_bra]*FAC12[orb_ket];

        double overlap2_g_pre0 {0.};
        for(int gaussC_bra = 0; gaussC_bra < nG_bra; gaussC_bra++){ //Iterate over the contracted Gaussians in the bra orbital
            double exponent_bra {orbitals_info_real[orb_bra][2*gaussC_bra + 3]};
            //double d_bra {orbitals_info_real[orb_bra][2*gaussC_bra + 4]};

            for(int gaussC_ket = 0; gaussC_ket < nG_ket; gaussC_ket++){ //Iterate over the contracted Gaussians in the ket orbital
                double exponent_ket {orbitals_info_real[orb_ket][2*gaussC_ket + 3]};
                //double d_ket {orbitals_info_real[orb_ket][2*gaussC_ket + 4]};

                double p {exponent_bra + exponent_ket};  //Exponent coefficient of the Hermite Gaussian
                arma::colvec P {(exponent_bra*coords_bra + exponent_ket*coords_ket)/p};  //Center of the Hermite Gaussian
                double PAx {P(0) - coords_bra(0)}; 
                double PAy {P(1) - coords_bra(1)}; 
                double PAz {P(2) - coords_bra(2)}; 
                double PBx {P(0) - coords_ket(0)}; 
                double PBy {P(1) - coords_ket(1)}; 
                double PBz {P(2) - coords_ket(2)}; 

                double overlap2_g_pre1 {0.};
                std::vector<int>::iterator g_itr_bra {g_coefs_bra.begin()};
                for(int numg_bra = 0; numg_bra < g_coefs_bra[0]; numg_bra++){ //Iterate over the summands of the corresponding spherical harmonic in the bra orbital
                    int i_bra {*(++g_itr_bra)};
                    int j_bra {*(++g_itr_bra)};
                    int k_bra {*(++g_itr_bra)};
                    int g_bra {*(++g_itr_bra)};
                    int Ei_bra {(i_bra*(i_bra + 1))/2};
                    int Ej_bra {(j_bra*(j_bra + 1))/2};
                    int Ek_bra {(k_bra*(k_bra + 1))/2};
                    
                    std::vector<int>::iterator g_itr_ket {g_coefs_ket.begin()};
                    for(int numg_ket = 0; numg_ket < g_coefs_ket[0]; numg_ket++){ //Iterate over the summands of the corresponding spherical harmonic in the ket orbital
                        int i_ket {*(++g_itr_ket)};
                        int j_ket {*(++g_itr_ket)};
                        int k_ket {*(++g_itr_ket)};
                        int g_ket {*(++g_itr_ket)};

                        double Eii0 {(i_bra >= i_ket)? Efunt0(i_ket + Ei_bra, p, PAx, PBx) : Efunt0(i_bra + (i_ket*(i_ket + 1))/2, p, PBx, PAx)};
                        double Ejj0 {(j_bra >= j_ket)? Efunt0(j_ket + Ej_bra, p, PAy, PBy) : Efunt0(j_bra + (j_ket*(j_ket + 1))/2, p, PBy, PAy)};
                        double Ekk0 {(k_bra >= k_ket)? Efunt0(k_ket + Ek_bra, p, PAz, PBz) : Efunt0(k_bra + (k_ket*(k_ket + 1))/2, p, PBz, PAz)};

                        overlap2_g_pre1 += g_bra*g_ket*Eii0*Ejj0*Ekk0;

                    }
                }
                overlap2_g_pre1 *= FAC3[orb_bra][gaussC_bra]*FAC3[orb_ket][gaussC_ket]*std::pow(p,-1.5)*std::exp(-exponent_bra*exponent_ket*norm_braket/p);
                overlap2_g_pre0 += overlap2_g_pre1;
            }
        }
        overlap2_g_pre0 *= FAC12_braket*PIpow;
        if(std::abs(overlap2_g_pre0) > etol){
            std::array<double,4> conv_vec = {overlap2_g_pre0,(double)orb_bra,(double)orb_ket,(double)Rind};
            overlap2Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

MPI_Barrier (MPI_COMM_WORLD);
// Store the matrices and the list of direct lattice vectors
uint64_t n_entries_thisproc = overlap2Matrices.size();
uint64_t n_entries_total;
MPI_Reduce(&n_entries_thisproc, &n_entries_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

std::string o2Cstr = IntFiles_Dir + "o2Mat_" + intName + ".o2c";
if(procMPI_rank == 0){
    std::ofstream output_file0(o2Cstr, std::ios::trunc);
    output_file0.close();
    std::ofstream output_file(o2Cstr, std::ios::app);
    output_file << "2-CENTER OVERLAP INTEGRALS" << std::endl;
    output_file << "Requested nR: " << nR << ". Computed nR: " << nR_star << std::endl;
    output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries_total/total_elem)*100 << " %" << std::endl;
    output_file << "Entry, mu, mu', R" << std::endl;
    output_file << n_entries_total << std::endl;
    output_file << dimMat_AUX << std::endl;
    output_file.close();
}
MPI_Barrier (MPI_COMM_WORLD);
for(int r = 0; r < procMPI_size; r++){
    if(procMPI_rank == r){
        std::ofstream output_file(o2Cstr, std::ios::app);
        output_file.precision(12);
        output_file << std::scientific;
        for(uint64_t ent = 0; ent < n_entries_thisproc; ent++){
            output_file << overlap2Matrices[ent][0] << "  " << static_cast<uint32_t>(overlap2Matrices[ent][1]) << " " << 
                static_cast<uint32_t>(overlap2Matrices[ent][2]) << " " << static_cast<uint32_t>(overlap2Matrices[ent][3]) << std::endl;
        }
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
}
if(procMPI_rank == 0){
    combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".o2c", arma::arma_ascii);
    std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
    std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "o2Mat_" + intName + ".o2c" << 
        " , and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".o2c" << std::endl;
}

}

}