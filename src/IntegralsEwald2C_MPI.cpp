#include "xatu/IntegralsEwald2C_MPI.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param scalei_supercell Vector where each component is the scaling factor for the corresponding original (unit cell) Bravais basis vectors Ri 
 *        to form the supercell. 
 * @param nR Minimum number of external supercell lattice vectors (i.e. repetitions of the supercell) to be included in the Ewald direct lattice sum.
 * @param nG Minimum number of reciprocal supercell vectors (i.e. reciprocal vectors for the lattice defined by the supercell) to be included in the Ewald reciprocal lattice sum.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .E2cDk instead of the usual .E2c 
 * @param intName Name of the file where the 2-center Ewald matrices will be stored as a vector (E2Mat_intName.E2c).
 */
IntegralsEwald2C_MPI::IntegralsEwald2C_MPI(const IntegralsBase& IntBase, const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const bool is_for_Dk, const std::string& intName) 
    : IntegralsBase{IntBase} {

    setgamma0(scalei_supercell);
    Ewald2Cfun(procMPI_rank, procMPI_size, tol, scalei_supercell, nR, nG, is_for_Dk, intName);

}

/**
 * Method to compute the Ewald matrices in the auxiliary basis (<P,0|A|P',R>) for a set of Bravais vectors within a supercell defined by scale
 * factors scalei_supercell (conmensurate with the k-grid for the BSE). Both direct and reciprocal lattice sums must be performed in the Ewald
 * potential, with a minimum number of terms nR and nG (respectively), in a lattice where the supercell is the unit of periodicity.
 * Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
 * indices: value,mu,mu',R,; in that order. The vector is saved in the E2Mat_intName.E2c file, and the list of supercell Bravais 
 * vectors in fractional coordinates is saved in the RlistFrac_intName.E2c file. Only the lower triangle of each R-matrix is stored; 
 * the upper triangle is given by hermiticity in the k-matrix
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param scalei_supercell Vector where each component is the scaling factor for the corresponding original (unit cell) Bravais basis vectors Ri 
 *        to form the supercell. 
 * @param nR Minimum number of external supercell lattice vectors (i.e. repetitions of the supercell) to be included in the Ewald direct lattice sum.
 * @param nG Minimum number of reciprocal supercell vectors (i.e. reciprocal vectors for the lattice defined by the supercell) to be included in the Ewald reciprocal lattice sum.
 * @param is_for_Dk If true, the integrals and lattice vectores will be stored with the extension .E2cDk instead of the usual .E2c 
 * @param intName Name of the file where the 2-center Ewald matrices will be stored as a vector (E2Mat_intName.E2c).
 * @return void. Matrices and the corresponding list of (inner) supercell lattice vectors are stored instead.
 */
void IntegralsEwald2C_MPI::Ewald2Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const bool is_for_Dk, const std::string& intName){

int32_t n1 = 0;
int32_t n2 = 0;
int32_t n3 = 0;
unify_ni(scalei_supercell,n1,n2,n3);

double PIpow1 = 0;
if(ndim == 2){
    PIpow1 = std::pow(PI,1.5) / (unitCellVolume*ANG2AU*ANG2AU*n1*n2);
} 
else if(ndim == 3){
    PIpow1 = std::pow(PI,1.5) / (unitCellVolume*ANG2AU*ANG2AU*ANG2AU*n1*n2*n3);
}
const double PIpow2 = std::pow(PI,2.5) * 2;
arma::mat combs_inner;
arma::mat RlistAU_inner   = ANG2AU*generateRlist_fixed(scalei_supercell, combs_inner, "Ewald2C", procMPI_rank); // lattice vectors within the supercell 
uint32_t nR_inner         = RlistAU_inner.n_cols;
this->RlistAU_outer_      = ANG2AU*generateRlist_supercell(nR, scalei_supercell, procMPI_rank);            // lattice vectors for supercell repetition
this->nR_outer_           = RlistAU_outer.n_cols;
this->GlistAU_half_       = (1./ANG2AU)*generateGlist_supercell_half(nG, scalei_supercell, procMPI_rank);  // reciprocal vectors with (reduced) supercell BZ
this->nG_                 = GlistAU_half.n_cols;
this->GlistAU_half_norms_ = arma::sqrt(arma::sum(GlistAU_half % GlistAU_half));

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat_AUX*(dimMat_AUX + 1))/2;
uint64_t total_elem = nelem_triang*nR_inner;

uint64_t elems_per_proc = total_elem / procMPI_size;
uint elems_remainder = total_elem % procMPI_size;
uint p1 = std::min(elems_remainder, static_cast<uint>(procMPI_rank));
uint p2 = std::min(elems_remainder, static_cast<uint>(procMPI_rank) + 1);

if(procMPI_rank == 0){
    std::cout << "Computing " << nR_inner << " " << dimMat_AUX << "x" << dimMat_AUX << " 2-center Ewald matrices in the AUX basis..." << std::flush;
}

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> Ewald2Matrices;
    Ewald2Matrices.reserve(elems_per_proc + 1);

    for(uint64_t s = procMPI_rank*elems_per_proc + p1; s < (procMPI_rank+1)*elems_per_proc + p2; s++){ //Spans the lower triangle of all the nR_inner matrices <P,0|A|P',R>
        uint64_t sind {s % nelem_triang};    //Index for the corresponding entry in the Ewald matrix, irrespective of the specific R
        uint32_t Rind {static_cast<uint32_t>(s / nelem_triang)};   //Position in RlistAU_inner (e.g. 0 for R=0) of the corresponding Bravais vector within the supercell
        uint64_t r = nelem_triang - (sind + 1);
        uint64_t l = (std::sqrt(8*r + 1) - 1)/2;
        uint32_t orb_bra = (sind + dimMat_AUX + (l*(l+1))/2 ) - nelem_triang;  //Orbital number (<dimMat) of the bra corresponding to the index s 
        uint32_t orb_ket = dimMat_AUX - (l + 1);                               //Orbital number (<dimMat) of the ket corresponding to the index s. orb_ket <= orb_bra (lower triangle)
        // arma::colvec R {RlistAU_inner.col(Rind)};  //Bravais vector (a.u.) within the supercell corresponding to the "s" matrix element

        int L_bra  {orbitals_info_int_AUX_[orb_bra][2]};
        int m_bra  {orbitals_info_int_AUX_[orb_bra][3]};
        int nG_bra {orbitals_info_int_AUX_[orb_bra][4]};
        arma::colvec coords_bra {orbitals_info_real_AUX_[orb_bra][0], orbitals_info_real_AUX_[orb_bra][1], orbitals_info_real_AUX_[orb_bra][2]};  //Position (a.u.) of bra atom
        std::vector<int> g_coefs_bra   {g_coefs_.at( L_bra*(L_bra + 1) + m_bra )};

        int L_ket  {orbitals_info_int_AUX_[orb_ket][2]};
        int m_ket  {orbitals_info_int_AUX_[orb_ket][3]};
        int nG_ket {orbitals_info_int_AUX_[orb_ket][4]};
        arma::colvec coords_ket {RlistAU_inner.col(Rind) + arma::colvec{orbitals_info_real_AUX_[orb_ket][0], orbitals_info_real_AUX_[orb_ket][1], orbitals_info_real_AUX_[orb_ket][2]} };  //Position (a.u.) of ket atom
        std::vector<int> g_coefs_ket   {g_coefs_.at( L_ket*(L_ket + 1) + m_ket )};

        arma::colvec coords_braket {coords_bra - coords_ket};
        double FAC12_braket = FAC12_AUX_[orb_bra]*FAC12_AUX_[orb_ket]; 

        double Ewald2_pre0 {0.};
        for(int gaussC_bra = 0; gaussC_bra < nG_bra; gaussC_bra++){ //Iterate over the contracted Gaussians in the bra orbital
            double exponent_bra {orbitals_info_real_AUX_[orb_bra][2*gaussC_bra + 3]};
            //double d_bra {orbitals_info_real_AUX[orb_bra][2*gaussC_bra + 4]};

            for(int gaussC_ket = 0; gaussC_ket < nG_ket; gaussC_ket++){ //Iterate over the contracted Gaussians in the ket orbital
                double exponent_ket {orbitals_info_real_AUX_[orb_ket][2*gaussC_ket + 3]};
                //double d_ket {orbitals_info_real_AUX[orb_ket][2*gaussC_ket + 4]};

                double p {exponent_bra + exponent_ket};  //Exponent coefficient of the Hermite Gaussian
                double mu {exponent_bra*exponent_ket/p};
                double gamma_fac {std::min(mu, gamma0_)};

                double Ewald2_pre1 {0.};
                std::vector<int>::iterator g_itr_bra {g_coefs_bra.begin()};
                for(int numg_bra = 0; numg_bra < g_coefs_bra[0]; numg_bra++){ //Iterate over the summands of the corresponding spherical harmonic in the bra orbital
                    int i_bra {*(++g_itr_bra)};
                    int j_bra {*(++g_itr_bra)};
                    int k_bra {*(++g_itr_bra)};
                    int g_bra {*(++g_itr_bra)};

                    arma::colvec Ei0vec_bra {Efun_single(i_bra, exponent_bra)};
                    arma::colvec Ej0vec_bra {Efun_single(j_bra, exponent_bra)};
                    arma::colvec Ek0vec_bra {Efun_single(k_bra, exponent_bra)};
                    
                    std::vector<int>::iterator g_itr_ket {g_coefs_ket.begin()};
                    for(int numg_ket = 0; numg_ket < g_coefs_ket[0]; numg_ket++){ //Iterate over the summands of the corresponding spherical harmonic in the ket orbital
                        int i_ket {*(++g_itr_ket)};
                        int j_ket {*(++g_itr_ket)};
                        int k_ket {*(++g_itr_ket)};
                        int g_ket {*(++g_itr_ket)};

                        arma::colvec Ei0vec_ket {Efun_single(i_ket, exponent_ket)};
                        arma::colvec Ej0vec_ket {Efun_single(j_ket, exponent_ket)};
                        arma::colvec Ek0vec_ket {Efun_single(k_ket, exponent_ket)};

                        for(int t_bra = i_bra; t_bra >= 0; t_bra -= 2){
                            for(int u_bra = j_bra; u_bra >= 0; u_bra -= 2){
                                for(int v_bra = k_bra; v_bra >= 0; v_bra -= 2){
                                    double Eijk_bra = Ei0vec_bra(t_bra)*Ej0vec_bra(u_bra)*Ek0vec_bra(v_bra);

                                    for(int t_ket = i_ket; t_ket >= 0; t_ket -= 2){
                                        for(int u_ket = j_ket; u_ket >= 0; u_ket -= 2){
                                            for(int v_ket = k_ket; v_ket >= 0; v_ket -= 2){
                                                double Eijk_ket {Ei0vec_ket(t_ket)*Ej0vec_ket(u_ket)*Ek0vec_ket(v_ket)};
                                                double sign_ket {(((t_ket + u_ket + v_ket) % 2) == 0)? 1. : -1.};
                                                int t_tot {t_bra + t_ket};
                                                int u_tot {u_bra + u_ket};
                                                int v_tot {v_bra + v_ket};

                                                // Direct lattice term
                                                double Ewald2_pre2direct = (mu > gamma0_)? Ewald2Cdirect(coords_braket, t_tot, u_tot, v_tot, mu) : 0.;

                                                // Reciprocal lattice term 
                                                double Ewald2_pre2recip = Ewald2Creciprocal(coords_braket, t_tot, u_tot, v_tot, gamma_fac);
                                                Ewald2_pre2recip *= PIpow1;    

                                                // Combine both lattice contributions
                                                Ewald2_pre1 += g_bra*g_ket*Eijk_bra*Eijk_ket * sign_ket * (Ewald2_pre2direct + Ewald2_pre2recip);
                                            }
                                        }
                                    }

                                }
                            }
                        }

                    }
                }
                Ewald2_pre1 *= FAC3_AUX_[orb_bra][gaussC_bra]*FAC3_AUX_[orb_ket][gaussC_ket]*std::pow(exponent_bra*exponent_ket,-1.5);
                Ewald2_pre0 += Ewald2_pre1;

            }
        }
        Ewald2_pre0 *= FAC12_braket*PIpow2;
        if(std::abs(Ewald2_pre0) > etol){
            std::array<double,4> conv_vec = {Ewald2_pre0,(double)orb_bra,(double)orb_ket,(double)Rind};
            Ewald2Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

MPI_Barrier (MPI_COMM_WORLD);
// Store the matrices and the list of direct lattice vectors
uint64_t n_entries_thisproc = Ewald2Matrices.size();
uint64_t n_entries_total;
MPI_Reduce(&n_entries_thisproc, &n_entries_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

std::string E2Cstr = is_for_Dk? IntFiles_Dir + "E2Mat_" + intName + ".E2cDk" : IntFiles_Dir + "E2Mat_" + intName + ".E2c";
if(procMPI_rank == 0){
    std::ofstream output_file0(E2Cstr, std::ios::trunc);
    output_file0.close();
    std::ofstream output_file(E2Cstr, std::ios::app);
    output_file << "2-CENTER EWALD INTEGRALS (" << ndim << "D)" << std::endl;
    output_file << "Supercell (inner) scaling: ";
    for(int ni = 0; ni < ndim; ni++){
        output_file << scalei_supercell[ni] << " "; 
    }
    output_file << std::endl;
    output_file << "Supercell (outer) sums: nR = " << nR_outer_ << ", nG = " << nG_ << std::endl; 
    output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries_total/total_elem)*100 << " %" << std::endl;
    output_file << "Entry, mu, mu', R" << std::endl;
    output_file << n_entries_total << std::endl;
    output_file << dimMat_AUX << std::endl;
    output_file.close();
}
MPI_Barrier (MPI_COMM_WORLD);
for(int r = 0; r < procMPI_size; r++){
    if(procMPI_rank == r){
        std::ofstream output_file(E2Cstr, std::ios::app);
        output_file.precision(12);
        output_file << std::scientific;
        for(uint64_t ent = 0; ent < n_entries_thisproc; ent++){
            output_file << Ewald2Matrices[ent][0] << "  " << static_cast<uint32_t>(Ewald2Matrices[ent][1]) << " " << 
            static_cast<uint32_t>(Ewald2Matrices[ent][2]) << " " << static_cast<uint32_t>(Ewald2Matrices[ent][3]) << std::endl;
        }
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
}
if(procMPI_rank == 0){
    std::string RE2Cstr = is_for_Dk? IntFiles_Dir + "RlistFrac_" + intName + ".E2cDk" : IntFiles_Dir + "RlistFrac_" + intName + ".E2c";
    combs_inner.save(RE2Cstr, arma::arma_ascii);
    std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
    std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << E2Cstr << 
        " , and list of Bravais vectors (within supercell) in " << RE2Cstr << std::endl;
}

}

}