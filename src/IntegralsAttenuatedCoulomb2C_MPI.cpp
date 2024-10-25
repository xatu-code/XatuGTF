#include "xatu/IntegralsAttenuatedCoulomb2C_MPI.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param omega Attenuation parameter in erfc(omega*|r-r'|), in atomic units (length^-1).
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed.
 * @param intName Name of the file where the 2-center attenuated Coulomb matrices will be stored as a vector (att0C2Mat_intName.att0C2c).
 */
IntegralsAttenuatedCoulomb2C_MPI::IntegralsAttenuatedCoulomb2C_MPI(const IntegralsBase& IntBase, const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR, const std::string& intName) : IntegralsBase{IntBase} {

    AttenuatedCoulomb2Cfun(procMPI_rank, procMPI_size, omega, tol, nR, intName);

}

/**
 * Method to compute the attenuated Coulomb matrices in the auxiliary basis (<P,0|erfc(wr)V_c|P',R>) for the first nR Bravais vectors R. 
 * These first nR (at least, until the star of vectors is completed) are generated with Lattice::generateRlist.
 * Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
 * indices: value,mu,mu',R,; in that order. The vector is saved in the att0C2Mat_intName.att0C2c file, and the list of Bravais 
 * vectors in fractional coordinates is saved in the RlistFrac_intName.att0C2c file. Only the lower triangle of each R-matrix is stored; 
 * the upper triangle is given by hermiticity in the k-matrix
 * @param omega Attenuation parameter in erfc(omega*|r-r'|), in atomic units (length^-1).
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed.
 * @param intName Name of the file where the 2-center attenuated Coulomb matrices will be stored as a vector (att0C2Mat_intName.att0C2c).
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsAttenuatedCoulomb2C_MPI::AttenuatedCoulomb2Cfun(const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR, const std::string& intName){

const double omega2 = omega*omega;
const double PIpow = std::pow(PI,2.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR, combs, "Att0Coulomb2C", procMPI_rank);  //convert Bravais vectors from Angstrom to atomic units
uint32_t nR_star = RlistAU.n_cols;

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat_AUX*(dimMat_AUX + 1))/2;
uint64_t total_elem = nelem_triang*nR_star;

uint64_t elems_per_proc = total_elem / procMPI_size;
uint elems_remainder = total_elem % procMPI_size;
uint p1 = std::min(elems_remainder, static_cast<uint>(procMPI_rank));
uint p2 = std::min(elems_remainder, static_cast<uint>(procMPI_rank) + 1);

if(procMPI_rank == 0){
    std::cout << "Computing " << nR_star << " " << dimMat_AUX << "x" << dimMat_AUX << " 2-center attenuated Coulomb matrices in the AUX basis..." << std::flush;
}

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> AttCoulomb2Matrices;
    AttCoulomb2Matrices.reserve(elems_per_proc + 1);

    for(uint64_t s = procMPI_rank*elems_per_proc + p1; s < (procMPI_rank+1)*elems_per_proc + p2; s++){ //Spans the lower triangle of all the nR_star matrices <P,0|V_c|P',R>
        uint64_t sind {s % nelem_triang};    //Index for the corresponding entry in the Coulomb matrix, irrespective of the specific R
        uint32_t Rind {static_cast<uint32_t>(s / nelem_triang)};         //Position in RlistAU (e.g. 0 for R=0) of the corresponding Bravais vector 
        uint64_t r = nelem_triang - (sind + 1);
        uint64_t l = (std::sqrt(8*r + 1) - 1)/2;
        uint32_t orb_bra = (sind + dimMat_AUX + (l*(l+1))/2 ) - nelem_triang;  //Orbital number (<dimMat) of the bra corresponding to the index s 
        uint32_t orb_ket = dimMat_AUX - (l + 1);                               //Orbital number (<dimMat) of the ket corresponding to the index s. orb_ket <= orb_bra (lower triangle)
        // arma::colvec R {RlistAU.col(Rind)};  //Bravais vector (a.u.) corresponding to the "s" matrix element

        int L_bra  {orbitals_info_int_AUX_[orb_bra][2]};
        int m_bra  {orbitals_info_int_AUX_[orb_bra][3]};
        int nG_bra {orbitals_info_int_AUX_[orb_bra][4]};
        arma::colvec coords_bra {orbitals_info_real_AUX_[orb_bra][0], orbitals_info_real_AUX_[orb_bra][1], orbitals_info_real_AUX_[orb_bra][2]};  //Position (a.u.) of bra atom
        std::vector<int> g_coefs_bra   {g_coefs_.at( L_bra*(L_bra + 1) + m_bra )};

        int L_ket  {orbitals_info_int_AUX_[orb_ket][2]};
        int m_ket  {orbitals_info_int_AUX_[orb_ket][3]};
        int nG_ket {orbitals_info_int_AUX_[orb_ket][4]};
        arma::colvec coords_ket {RlistAU.col(Rind) + arma::colvec{orbitals_info_real_AUX_[orb_ket][0], orbitals_info_real_AUX_[orb_ket][1], orbitals_info_real_AUX_[orb_ket][2]} };  //Position (a.u.) of ket atom
        std::vector<int> g_coefs_ket   {g_coefs_.at( L_ket*(L_ket + 1) + m_ket )};

        arma::colvec coords_braket {coords_bra - coords_ket};
        double FAC12_braket = FAC12_AUX_[orb_bra]*FAC12_AUX_[orb_ket]; 

        double AttCoulomb2_pre0 {0.};
        for(int gaussC_bra = 0; gaussC_bra < nG_bra; gaussC_bra++){ //Iterate over the contracted Gaussians in the bra orbital
            double exponent_bra {orbitals_info_real_AUX_[orb_bra][2*gaussC_bra + 3]};
            //double d_bra {orbitals_info_real_AUX[orb_bra][2*gaussC_bra + 4]};

            for(int gaussC_ket = 0; gaussC_ket < nG_ket; gaussC_ket++){ //Iterate over the contracted Gaussians in the ket orbital
                double exponent_ket {orbitals_info_real_AUX_[orb_ket][2*gaussC_ket + 3]};
                //double d_ket {orbitals_info_real_AUX[orb_ket][2*gaussC_ket + 4]};

                double p {exponent_bra + exponent_ket};  //Exponent coefficient of the Hermite Gaussian
                double mu {exponent_bra*exponent_ket/p};
                double mu_omega {mu*omega2/(mu + omega2)};
                double mu_sqrt {std::sqrt(mu)};
                double mu_omega_sqrt {std::sqrt(mu_omega)};

                double AttCoulomb2_pre1 {0.};
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
                                                double sign_ket {(((t_ket + u_ket + v_ket) % 2) == 0)? 1. : -1.};
                                                int t_tot = t_bra + t_ket;
                                                int u_tot = u_bra + u_ket;
                                                int v_tot = v_bra + v_ket;
                                                double Hermit1, Hermit2;
                                                if(t_tot >= u_tot){
                                                    if(u_tot >= v_tot){ // (t,u,v)
                                                        Hermit1 = HermiteCoulomb(t_tot, u_tot, v_tot, mu,       coords_braket(0), coords_braket(1), coords_braket(2));
                                                        Hermit2 = HermiteCoulomb(t_tot, u_tot, v_tot, mu_omega, coords_braket(0), coords_braket(1), coords_braket(2));
                                                    }
                                                    else if(t_tot >= v_tot){ // (t,v,u)
                                                        Hermit1 = HermiteCoulomb(t_tot, v_tot, u_tot, mu,       coords_braket(0), coords_braket(2), coords_braket(1));
                                                        Hermit2 = HermiteCoulomb(t_tot, v_tot, u_tot, mu_omega, coords_braket(0), coords_braket(2), coords_braket(1));
                                                    }
                                                    else{ // (v,t,u)
                                                        Hermit1 = HermiteCoulomb(v_tot, t_tot, u_tot, mu,       coords_braket(2), coords_braket(0), coords_braket(1));
                                                        Hermit2 = HermiteCoulomb(v_tot, t_tot, u_tot, mu_omega, coords_braket(2), coords_braket(0), coords_braket(1));
                                                    }
                                                } 
                                                else if(u_tot >= v_tot){ 
                                                    if(t_tot >= v_tot){ // (u,t,v)
                                                        Hermit1 = HermiteCoulomb(u_tot, t_tot, v_tot, mu,       coords_braket(1), coords_braket(0), coords_braket(2));
                                                        Hermit2 = HermiteCoulomb(u_tot, t_tot, v_tot, mu_omega, coords_braket(1), coords_braket(0), coords_braket(2));
                                                    }
                                                    else{ // (u,v,t)
                                                        Hermit1 = HermiteCoulomb(u_tot, v_tot, t_tot, mu,       coords_braket(1), coords_braket(2), coords_braket(0));
                                                        Hermit2 = HermiteCoulomb(u_tot, v_tot, t_tot, mu_omega, coords_braket(1), coords_braket(2), coords_braket(0));
                                                    }
                                                }
                                                else{ // (v,u,t)
                                                    Hermit1 = HermiteCoulomb(v_tot, u_tot, t_tot, mu,       coords_braket(2), coords_braket(1), coords_braket(0));
                                                    Hermit2 = HermiteCoulomb(v_tot, u_tot, t_tot, mu_omega, coords_braket(2), coords_braket(1), coords_braket(0));
                                                }
                                                
                                                AttCoulomb2_pre1 += g_bra*g_ket*sign_ket*Eijk_bra*Ei0vec_ket(t_ket)*Ej0vec_ket(u_ket)*Ek0vec_ket(v_ket)*(mu_sqrt*Hermit1 - mu_omega_sqrt*Hermit2);

                                            }
                                        }
                                    }

                                }
                            }
                        }


                    }
                }
                AttCoulomb2_pre1 *= FAC3_AUX_[orb_bra][gaussC_bra]*FAC3_AUX_[orb_ket][gaussC_ket]*std::pow(exponent_bra*exponent_ket,-1.5);
                AttCoulomb2_pre0 += AttCoulomb2_pre1;

            }
        }
        AttCoulomb2_pre0 *= FAC12_braket*2*PIpow;
        if(std::abs(AttCoulomb2_pre0) > etol){
            std::array<double,4> conv_vec = {AttCoulomb2_pre0,(double)orb_bra,(double)orb_ket,(double)Rind};
            AttCoulomb2Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

MPI_Barrier (MPI_COMM_WORLD);
// Store the matrices and the list of direct lattice vectors
uint64_t n_entries_thisproc = AttCoulomb2Matrices.size();
uint64_t n_entries_total;
MPI_Reduce(&n_entries_thisproc, &n_entries_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

std::string att0C2Cstr = IntFiles_Dir + "att0C2Mat_" + intName + ".att0C2c";
if(procMPI_rank == 0){
    std::ofstream output_file0(att0C2Cstr, std::ios::trunc);
    output_file0.close();
    std::ofstream output_file(att0C2Cstr, std::ios::app);
    output_file << "2-CENTER ATTENUATED CAP(0) COULOMB INTEGRALS" << std::endl;
    output_file << "Requested nR: " << nR << ". Computed nR: " << nR_star << ". omega (a.u.) = " << omega << std::endl;
    output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries_total/total_elem)*100 << " %" << std::endl;
    output_file << "Entry, mu, mu', R" << std::endl;
    output_file << n_entries_total << std::endl;
    output_file << dimMat_AUX << std::endl;
    output_file.close();
}
MPI_Barrier (MPI_COMM_WORLD);
for(int r = 0; r < procMPI_size; r++){
    if(procMPI_rank == r){
        std::ofstream output_file(att0C2Cstr, std::ios::app);
        output_file.precision(12);
        output_file << std::scientific;
        for(uint64_t ent = 0; ent < n_entries_thisproc; ent++){
            output_file << AttCoulomb2Matrices[ent][0] << "  " << static_cast<uint32_t>(AttCoulomb2Matrices[ent][1]) << " " << 
                static_cast<uint32_t>(AttCoulomb2Matrices[ent][2]) << " " << static_cast<uint32_t>(AttCoulomb2Matrices[ent][3]) << std::endl;
        }
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
}
if(procMPI_rank == 0){
    combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".att0C2c", arma::arma_ascii);
    std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
    std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << att0C2Cstr << 
        " , and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".att0C2c" << std::endl;
}

}

}