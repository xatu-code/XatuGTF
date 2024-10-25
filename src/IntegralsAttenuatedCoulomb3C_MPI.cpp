#include "xatu/IntegralsAttenuatedCoulomb3C_MPI.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param omega Attenuation parameter in erfc(omega*|r-r'|), in atomic units (length^-1).
 * @param tol Tolerance for retaining the entries of the 3-center attenuated Coulomb matrices. These must be > 10^-tol, in absolute value.
 * @param nR2 Number of R and R' that will be considered for the integrals. 
 * @param intName Name of the file where the 3-center attenuated Coulomb matrices will be stored as a vector (att0C3Mat_intName.att0C3c).
 */
IntegralsAttenuatedCoulomb3C_MPI::IntegralsAttenuatedCoulomb3C_MPI(const IntegralsBase& IntBase, const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR2, const std::string& intName) : IntegralsBase{IntBase} {

    AttenuatedCoulomb3Cfun(procMPI_rank, procMPI_size, omega, tol, nR2, intName);

}

/**
 * Method to compute the rectangular attenuated Coulomb matrices <P,0|erfc(wr)V_c|mu,R;mu',R'> in the mixed SCF and auxiliary basis sets for the 
 * first nR2 Bravais vectors R and R' (nR2^2 pairs of vectors). These first nR (at least, until the star of vectors is completed)
 * are generated with IntegralsBase::generateRlist. Each entry above a certain tolerance (10^-tol) is stored in an entry of a 
 * vector (of arrays) along with the corresponding indices: value,P,mu,mu',R,R'; in that order. The vector is saved in the 
 * att0C3Mat_intName.att0C3c file. The list of Bravais vectors (for a single index) in fractional coordinates is saved in the RlistFrac_intName.att0C3c file
 * @param omega Attenuation parameter in erfc(omega*|r-r'|), in atomic units (length^-1).
 * @param nR2 Square root of the minimum number of direct lattice vectors for which the 3-center attenuated Coulomb integrals will be computed.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param intName Name of the file where the 3-center attenuated Coulomb matrices will be stored as a vector (att0C3Mat_intName.att0C3c).
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsAttenuatedCoulomb3C_MPI::AttenuatedCoulomb3Cfun(const int procMPI_rank, const int procMPI_size, const double omega, const int tol, const uint32_t nR2, const std::string& intName){

const double omega2 = omega*omega;
const double PIpow = std::pow(PI,2.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR2, combs, "Att0Coulomb3C", procMPI_rank);
uint32_t nR2_star = RlistAU.n_cols;

double etol = std::pow(10.,-tol);
uint64_t dim_Slice = dimMat_SCF*nR2_star;
uint64_t nelems_slice = dim_Slice*dim_Slice;
uint64_t total_elem = nelems_slice*dimMat_AUX;

uint64_t elems_per_proc = total_elem / procMPI_size;
uint elems_remainder = total_elem % procMPI_size;
uint p1 = std::min(elems_remainder, static_cast<uint>(procMPI_rank));
uint p2 = std::min(elems_remainder, static_cast<uint>(procMPI_rank) + 1);

if(procMPI_rank == 0){
    std::cout << "Computing " << dimMat_AUX << " " << dimMat_SCF << "x" << nR2_star << "x" << dimMat_SCF << "x" << nR2_star << " 3-center attenuated Coulomb matrices..." << std::flush;
}


// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,6>> AttCoulomb3Matrices;
    AttCoulomb3Matrices.reserve((elems_per_proc + 1)/3);

    for(uint64_t s = procMPI_rank*elems_per_proc + p1; s < (procMPI_rank+1)*elems_per_proc + p2; s++){ //Spans all the matrix elements 
        uint64_t sind {s % nelems_slice};   //Index for the entry within the slice matrix (single index for {mu,R,mu',R'}) corresponding to loop index s. Iterates first over columns (mu,R)
        
        uint32_t Pind {static_cast<uint32_t>(s / nelems_slice)};     //Slice of the cube (or AUX basis component P, <dimMat_AUX) corresponding to loop index s 
        uint64_t muRind1 {sind % dim_Slice};  //Row in the slice matrix (single index for {mu,R})
        uint32_t muind1 {static_cast<uint32_t>(muRind1 % dimMat_SCF_)};    //Orbital number mu (<dimMat_SCF) corresponding to loop index s
        uint32_t Rind1  {static_cast<uint32_t>(muRind1 / dimMat_SCF_)};    //Direct lattice vector index (<nR2_star) corresponding to loop index s
        uint64_t muRind2 {sind / dim_Slice};  //Column in the slice matrix (single index for {mu',R'}) 
        uint32_t muind2 {static_cast<uint32_t>(muRind2 % dimMat_SCF_)};    //Orbital number mu' (<dimMat_SCF) corresponding to loop index s 
        uint32_t Rind2  {static_cast<uint32_t>(muRind2 / dimMat_SCF_)};    //Direct lattice vector R' (<nR2_star) corresponding to loop index s

        int L_P  {orbitals_info_int_AUX_[Pind][2]};
        int m_P  {orbitals_info_int_AUX_[Pind][3]};
        int nG_P {orbitals_info_int_AUX_[Pind][4]};
        arma::colvec coords_P {orbitals_info_real_AUX_[Pind][0], orbitals_info_real_AUX_[Pind][1], orbitals_info_real_AUX_[Pind][2]};  //Position (a.u.) of P atom
        std::vector<int> g_coefs_P {g_coefs_.at( L_P*(L_P + 1) + m_P )};

        int L_mu1  {orbitals_info_int_SCF_[muind1][2]};
        int m_mu1  {orbitals_info_int_SCF_[muind1][3]};
        int nG_mu1 {orbitals_info_int_SCF_[muind1][4]};
        // arma::colvec R1 {RlistAU.col(Rind1)};  //Direct lattice vector R corresponding to loop index s
        arma::colvec coords_mu1 {RlistAU.col(Rind1) + arma::colvec{orbitals_info_real_SCF_[muind1][0], orbitals_info_real_SCF_[muind1][1], orbitals_info_real_SCF_[muind1][2]} };  //Position (a.u.) of mu atom
        std::vector<int> g_coefs_mu1 {g_coefs_.at( L_mu1*(L_mu1 + 1) + m_mu1 )};

        int L_mu2  {orbitals_info_int_SCF_[muind2][2]};
        int m_mu2  {orbitals_info_int_SCF_[muind2][3]};
        int nG_mu2 {orbitals_info_int_SCF_[muind2][4]};
        // arma::colvec R2 {RlistAU.col(Rind2)};  //Direct lattice vector R corresponding to loop index s
        arma::colvec coords_mu2 {RlistAU.col(Rind2) + arma::colvec{orbitals_info_real_SCF_[muind2][0], orbitals_info_real_SCF_[muind2][1], orbitals_info_real_SCF_[muind2][2]} };  //Position (a.u.) of mu' atom
        std::vector<int> g_coefs_mu2 {g_coefs_.at( L_mu2*(L_mu2 + 1) + m_mu2 )};

        double norm_mu1mu2 {arma::dot(coords_mu1 - coords_mu2, coords_mu1 - coords_mu2)};
        double AttCoulomb3_pre0 {0.};
        for(int gaussC_mu1 = 0; gaussC_mu1 < nG_mu1; gaussC_mu1++){ //Iterate over the contracted Gaussians in the mu orbital 
            double exponent_mu1 {orbitals_info_real_SCF_[muind1][2*gaussC_mu1 + 3]};

            for(int gaussC_mu2 = 0; gaussC_mu2 < nG_mu2; gaussC_mu2++){ //Iterate over the contracted Gaussians in the mu' orbital
                double exponent_mu2 {orbitals_info_real_SCF_[muind2][2*gaussC_mu2 + 3]};
                double p {exponent_mu1 + exponent_mu2};  //Exponent coefficient of the Hermite Gaussian between mu,mu'
                arma::colvec P {(exponent_mu1*coords_mu1 + exponent_mu2*coords_mu2)/p};  //Center of the Hermite Gaussian between mu,mu'
                double PAx {P(0) - coords_mu1(0)}; 
                double PAy {P(1) - coords_mu1(1)}; 
                double PAz {P(2) - coords_mu1(2)}; 
                double PBx {P(0) - coords_mu2(0)}; 
                double PBy {P(1) - coords_mu2(1)}; 
                double PBz {P(2) - coords_mu2(2)};

                for(int gaussC_P = 0; gaussC_P < nG_P; gaussC_P++){ //Iterate over the contracted Gaussians in the P orbital
                    double exponent_P {orbitals_info_real_AUX_[Pind][2*gaussC_P + 3]};
                    double p_expP {p*exponent_P};
                    double p_3red {p_expP / (exponent_P + p)};
                    double p_3red_omega {p_3red*omega2 / (p_3red + omega2)};
                    double p_3red_sqrt {std::sqrt(p_3red)};
                    double p_3red_omega_sqrt {std::sqrt(p_3red_omega)};
                    arma::colvec coords_Pbraket {coords_P - P}; //Vector from the center of the Hermite Gaussian (mu,mu') to the center of AUX basis function P

                    double AttCoulomb3_pre1 {0.};
                    std::vector<int>::iterator g_itr_mu1 {g_coefs_mu1.begin()};
                    for(int numg_mu1 = 0; numg_mu1 < g_coefs_mu1[0]; numg_mu1++){ //Iterate over the summands of the corresponding spherical harmonic in the mu orbital
                        int i_mu1 {*(++g_itr_mu1)};
                        int j_mu1 {*(++g_itr_mu1)};
                        int k_mu1 {*(++g_itr_mu1)};
                        int g_mu1 {*(++g_itr_mu1)};
                        int Ei_mu1 {(i_mu1*(i_mu1 + 1))/2};
                        int Ej_mu1 {(j_mu1*(j_mu1 + 1))/2};
                        int Ek_mu1 {(k_mu1*(k_mu1 + 1))/2};

                        std::vector<int>::iterator g_itr_mu2 {g_coefs_mu2.begin()};
                        for(int numg_mu2 = 0; numg_mu2 < g_coefs_mu2[0]; numg_mu2++){ //Iterate over the summands of the corresponding spherical harmonic in the mu' orbital
                            int i_mu2 {*(++g_itr_mu2)};
                            int j_mu2 {*(++g_itr_mu2)};
                            int k_mu2 {*(++g_itr_mu2)};
                            int g_mu2 {*(++g_itr_mu2)};
                            int i_mu1mu2 {i_mu1 + i_mu2};
                            int j_mu1mu2 {j_mu1 + j_mu2};
                            int k_mu1mu2 {k_mu1 + k_mu2};
                            int Ei_mu2 {(i_mu2*(i_mu2 + 1))/2};
                            int Ej_mu2 {(j_mu2*(j_mu2 + 1))/2};
                            int Ek_mu2 {(k_mu2*(k_mu2 + 1))/2};

                            arma::colvec Eii_vec_mu {(i_mu1 >= i_mu2)? Efun(i_mu2 + Ei_mu1, p, PAx, PBx) : Efun(i_mu1 + Ei_mu2, p, PBx, PAx)};
                            arma::colvec Ejj_vec_mu {(j_mu1 >= j_mu2)? Efun(j_mu2 + Ej_mu1, p, PAy, PBy) : Efun(j_mu1 + Ej_mu2, p, PBy, PAy)};
                            arma::colvec Ekk_vec_mu {(k_mu1 >= k_mu2)? Efun(k_mu2 + Ek_mu1, p, PAz, PBz) : Efun(k_mu1 + Ek_mu2, p, PBz, PAz)};

                            std::vector<int>::iterator g_itr_P {g_coefs_P.begin()};
                            for(int numg_P = 0; numg_P < g_coefs_P[0]; numg_P++){ //Iterate over the summands of the corresponding spherical harmonic in the P orbital
                                int i_P {*(++g_itr_P)};
                                int j_P {*(++g_itr_P)};
                                int k_P {*(++g_itr_P)};
                                int g_P {*(++g_itr_P)};
                                arma::colvec Ei0vec_P {Efun_single(i_P, exponent_P)};
                                arma::colvec Ej0vec_P {Efun_single(j_P, exponent_P)};
                                arma::colvec Ek0vec_P {Efun_single(k_P, exponent_P)};

                                for(int t_mu = 0; t_mu <= i_mu1mu2; t_mu++){
                                    for(int u_mu = 0; u_mu <= j_mu1mu2; u_mu++){
                                        for(int v_mu = 0; v_mu <= k_mu1mu2; v_mu++){
                                            double sign_mu {(((t_mu + u_mu + v_mu) % 2) == 0)? 1. : -1.};
                                            double Eijk_mu = Eii_vec_mu(t_mu)*Ejj_vec_mu(u_mu)*Ekk_vec_mu(v_mu);

                                            for(int t_P = i_P; t_P >= 0; t_P -= 2){
                                                for(int u_P = j_P; u_P >= 0; u_P -= 2){
                                                    for(int v_P = k_P; v_P >= 0; v_P -= 2){
                                                        double Eijk_P = Ei0vec_P(t_P)*Ej0vec_P(u_P)*Ek0vec_P(v_P);
                                                        int t_tot = t_mu + t_P;
                                                        int u_tot = u_mu + u_P;
                                                        int v_tot = v_mu + v_P;
                                                        double Hermit1, Hermit2;

                                                        if(t_tot >= u_tot){
                                                            if(u_tot >= v_tot){ // (t,u,v)
                                                                Hermit1 = HermiteCoulomb(t_tot, u_tot, v_tot, p_3red,       coords_Pbraket(0), coords_Pbraket(1), coords_Pbraket(2));
                                                                Hermit2 = HermiteCoulomb(t_tot, u_tot, v_tot, p_3red_omega, coords_Pbraket(0), coords_Pbraket(1), coords_Pbraket(2));
                                                            }
                                                            else if(t_tot >= v_tot){ // (t,v,u)
                                                                Hermit1 = HermiteCoulomb(t_tot, v_tot, u_tot, p_3red,       coords_Pbraket(0), coords_Pbraket(2), coords_Pbraket(1));
                                                                Hermit2 = HermiteCoulomb(t_tot, v_tot, u_tot, p_3red_omega, coords_Pbraket(0), coords_Pbraket(2), coords_Pbraket(1));
                                                            }
                                                            else{ // (v,t,u)
                                                                Hermit1 = HermiteCoulomb(v_tot, t_tot, u_tot, p_3red,       coords_Pbraket(2), coords_Pbraket(0), coords_Pbraket(1));
                                                                Hermit2 = HermiteCoulomb(v_tot, t_tot, u_tot, p_3red_omega, coords_Pbraket(2), coords_Pbraket(0), coords_Pbraket(1));
                                                            }
                                                        } 
                                                        else if(u_tot >= v_tot){ 
                                                            if(t_tot >= v_tot){ // (u,t,v)
                                                                Hermit1 = HermiteCoulomb(u_tot, t_tot, v_tot, p_3red,       coords_Pbraket(1), coords_Pbraket(0), coords_Pbraket(2));
                                                                Hermit2 = HermiteCoulomb(u_tot, t_tot, v_tot, p_3red_omega, coords_Pbraket(1), coords_Pbraket(0), coords_Pbraket(2));
                                                            }
                                                            else{ // (u,v,t)
                                                                Hermit1 = HermiteCoulomb(u_tot, v_tot, t_tot, p_3red,       coords_Pbraket(1), coords_Pbraket(2), coords_Pbraket(0));
                                                                Hermit2 = HermiteCoulomb(u_tot, v_tot, t_tot, p_3red_omega, coords_Pbraket(1), coords_Pbraket(2), coords_Pbraket(0));
                                                            }
                                                        }
                                                        else{ // (v,u,t)
                                                            Hermit1 = HermiteCoulomb(v_tot, u_tot, t_tot, p_3red,       coords_Pbraket(2), coords_Pbraket(1), coords_Pbraket(0));
                                                            Hermit2 = HermiteCoulomb(v_tot, u_tot, t_tot, p_3red_omega, coords_Pbraket(2), coords_Pbraket(1), coords_Pbraket(0));
                                                        }
                                                        
                                                        AttCoulomb3_pre1 += g_P*g_mu1*g_mu2*Eijk_P*Eijk_mu*sign_mu*(p_3red_sqrt*Hermit1 - p_3red_omega_sqrt*Hermit2);

                                                    }
                                                }
                                            }

                                        }
                                    }
                                }

                            }
                        }
                    }
                    AttCoulomb3_pre1 *= FAC3_SCF_[muind1][gaussC_mu1]*FAC3_SCF_[muind2][gaussC_mu2]*FAC3_AUX_[Pind][gaussC_P]*std::pow(p_expP,-1.5)*std::exp(- exponent_mu1*exponent_mu2*norm_mu1mu2 / p);
                    AttCoulomb3_pre0 += AttCoulomb3_pre1;
                }
            }
        }

        AttCoulomb3_pre0 *= FAC12_AUX_[Pind]*FAC12_SCF_[muind1]*FAC12_SCF_[muind2]*2*PIpow;
        if(std::abs(AttCoulomb3_pre0) > etol){
            std::array<double,6> conv_vec = {AttCoulomb3_pre0,(double)Pind,(double)muind1,(double)muind2,(double)Rind1,(double)Rind2};
            AttCoulomb3Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

MPI_Barrier (MPI_COMM_WORLD);
// Store the matrices and the list of direct lattice vectors
uint64_t n_entries_thisproc = AttCoulomb3Matrices.size();
uint64_t n_entries_total;
MPI_Reduce(&n_entries_thisproc, &n_entries_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

std::string att0C3Cstr = IntFiles_Dir + "att0C3Mat_" + intName + ".att0C3c";
if(procMPI_rank == 0){
    std::ofstream output_file0(att0C3Cstr, std::ios::trunc);
    output_file0.close();
    std::ofstream output_file(att0C3Cstr, std::ios::app);
    output_file << "3-CENTER ATTENUATED CAP(0) COULOMB INTEGRALS" << std::endl;
    output_file << "Requested nR: " << nR2 << ". Computed nR: " << nR2_star << ". omega (a.u.) = " << omega << std::endl;
    output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries_total/total_elem)*100 << " %" << std::endl;
    output_file << "Entry, P, mu, mu', R, R'" << std::endl;
    output_file << n_entries_total << std::endl;
    output_file << dimMat_AUX << " " << dimMat_SCF << std::endl;
    output_file.close();
}
MPI_Barrier (MPI_COMM_WORLD);
for(int r = 0; r < procMPI_size; r++){
    if(procMPI_rank == r){
        std::ofstream output_file(att0C3Cstr, std::ios::app);
        output_file.precision(12);
        output_file << std::scientific;
        for(uint64_t ent = 0; ent < n_entries_thisproc; ent++){
            output_file << AttCoulomb3Matrices[ent][0] << "  " << static_cast<uint32_t>(AttCoulomb3Matrices[ent][1]) << " " << 
                static_cast<uint32_t>(AttCoulomb3Matrices[ent][2]) << " " << static_cast<uint32_t>(AttCoulomb3Matrices[ent][3]) << " " << 
                    static_cast<uint32_t>(AttCoulomb3Matrices[ent][4]) << " " << static_cast<uint32_t>(AttCoulomb3Matrices[ent][5]) << std::endl;
        }
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
}
if(procMPI_rank == 0){
    combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".att0C3c",arma::arma_ascii);
    std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
    std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << att0C3Cstr << 
        ", and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".att0C3c" << std::endl;
}

}

}