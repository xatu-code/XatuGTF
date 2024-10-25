#include "xatu/IntegralsOverlap3C_MPI.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param tol Tolerance for retaining the entries of the 3-center overlap matrices. These must be > 10^-tol, in absolute value.
 * @param nR2 Number of R and R' that will be considered for the integrals. 
 * @param intName Name of the file where the 3-center overlap matrices will be stored as a cube (o3Mat_intName.o3c).
 */
IntegralsOverlap3C_MPI::IntegralsOverlap3C_MPI(const IntegralsBase& IntBase, const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR2, const std::string& intName) : IntegralsBase{IntBase} {

    overlap3Cfun(procMPI_rank, procMPI_size, tol, nR2, intName);

}

/**
 * Method to compute the rectangular overlap matrices <P,0|mu,R;mu',R'> in the mixed SCF and auxiliary basis sets for the first nR Bravais
 * vectors R and R' (nR2^2 pairs of vectors). These first nR (at least, until the star of vectors is completed) are generated with IntegralsBase::generateRlist.
 * Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding indices:
 * value,P,mu,mu',R,R'; in that order. The vector is saved in the o3Mat_intName.o3c file. The list of Bravais vectors (for a single index) in fractional coordinates 
 * is saved in the RlistFrac_intName.o3c file.
 * @param nR2 Square root of the minimum number of direct lattice vectors for which the 3-center overlap integrals will be computed.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param intName Name of the file where the 3-center overlap matrices will be stored (o3Mat_intName.o3c).
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsOverlap3C_MPI::overlap3Cfun(const int procMPI_rank, const int procMPI_size, const int tol, const uint32_t nR2, const std::string& intName){

const double PIpow = std::pow(PI,1.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR2, combs, "Overlap3C", procMPI_rank);
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
    std::cout << "Computing " << dimMat_AUX << " " << dimMat_SCF << "x" << nR2_star << "x" << dimMat_SCF << "x" << nR2_star << " 3-center overlap matrices..." << std::flush;
}


// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,6>> overlap3Matrices;
    overlap3Matrices.reserve((elems_per_proc + 1)/4);

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

        double norm_mu1mu2 {arma::dot(coords_mu1 - coords_mu2,coords_mu1 - coords_mu2)};
        double overlap3_g_pre0 {0.};
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

                double norm_PP_3 {arma::dot(P - coords_P, P - coords_P)};
                for(int gaussC_P = 0; gaussC_P < nG_P; gaussC_P++){ //Iterate over the contracted Gaussians in the P orbital
                    double exponent_P {orbitals_info_real_AUX_[Pind][2*gaussC_P + 3]};

                    double p_3 {exponent_P + p};  //Exponent coefficient of the triple Hermite Gaussian (between P and resulting of the Hermite Gaussian mu,mu')
                    arma::colvec P_3 {(p*P + exponent_P*coords_P)/p_3};  //Center of the triple Hermite Gaussian
                    double PAx_3 {P_3(0) - P(0)};
                    double PAy_3 {P_3(1) - P(1)};
                    double PAz_3 {P_3(2) - P(2)};
                    double PBx_3 {P_3(0) - coords_P(0)};
                    double PBy_3 {P_3(1) - coords_P(1)};
                    double PBz_3 {P_3(2) - coords_P(2)};

                    double overlap3_g_pre1 {0.};
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
                            int Ei_mu2 {(i_mu2*(i_mu2 + 1))/2};
                            int Ej_mu2 {(j_mu2*(j_mu2 + 1))/2};
                            int Ek_mu2 {(k_mu2*(k_mu2 + 1))/2};

                            arma::colvec Eii_vec {(i_mu1 >= i_mu2)? Efun(i_mu2 + Ei_mu1, p, PAx, PBx) : Efun(i_mu1 + Ei_mu2, p, PBx, PAx)};
                            arma::colvec Ejj_vec {(j_mu1 >= j_mu2)? Efun(j_mu2 + Ej_mu1, p, PAy, PBy) : Efun(j_mu1 + Ej_mu2, p, PBy, PAy)};
                            arma::colvec Ekk_vec {(k_mu1 >= k_mu2)? Efun(k_mu2 + Ek_mu1, p, PAz, PBz) : Efun(k_mu1 + Ek_mu2, p, PBz, PAz)};

                            std::vector<int>::iterator g_itr_P {g_coefs_P.begin()};
                            for(int numg_P = 0; numg_P < g_coefs_P[0]; numg_P++){ //Iterate over the summands of the corresponding spherical harmonic in the P orbital
                                int i_P {*(++g_itr_P)};
                                int j_P {*(++g_itr_P)};
                                int k_P {*(++g_itr_P)};
                                int g_P {*(++g_itr_P)};

                                // X contribution
                                double sum_t {0.};
                                for(int t = 0; t <= (i_mu1 + i_mu2); t++){
                                    std::vector<double> Dt_vec {Dfun(t, p)};
                                    double sum_ipp {0.};
                                    double E3ii_0;
                                    for(int i_pp = t; i_pp >= 0; i_pp -= 2){
                                        if(i_pp <= 4){
                                            E3ii_0 = (i_pp >= i_P)? Efunt0(i_P + (i_pp*(i_pp+1))/2, p_3, PAx_3, PBx_3) : Efunt0(i_pp + (i_P*(i_P+1))/2, p_3, PBx_3, PAx_3);
                                        } else {
                                            E3ii_0 = (i_pp >= i_P)? EfunTriplet0(5*i_pp + i_P - 25, p_3, PAx_3, PBx_3) : EfunTriplet0(5*i_P + i_pp - 25, p_3, PBx_3, PAx_3);
                                        }
                                        sum_ipp += Dt_vec[i_pp/2]*E3ii_0;
                                    }
                                    sum_t += Eii_vec(t)*sum_ipp;
                                }   
                                // Y contribution
                                double sum_u {0.};
                                for(int u = 0; u <= (j_mu1 + j_mu2); u++){
                                    std::vector<double> Du_vec {Dfun(u, p)};
                                    double sum_jpp {0.};
                                    double E3jj_0;
                                    for(int j_pp = u; j_pp >= 0; j_pp -= 2){
                                        if(j_pp <= 4){
                                            E3jj_0 = (j_pp >= j_P)? Efunt0(j_P + (j_pp*(j_pp+1))/2, p_3, PAy_3, PBy_3) : Efunt0(j_pp + (j_P*(j_P+1))/2, p_3, PBy_3, PAy_3);
                                        } else {
                                            E3jj_0 = (j_pp >= j_P)? EfunTriplet0(5*j_pp + j_P - 25, p_3, PAy_3, PBy_3) : EfunTriplet0(5*j_P + j_pp - 25, p_3, PBy_3, PAy_3);
                                        }
                                        sum_jpp += Du_vec[j_pp/2]*E3jj_0;
                                    }
                                    sum_u += Ejj_vec(u)*sum_jpp;
                                }
                                // Z contribution
                                double sum_v {0.};
                                for(int v = 0; v <= (k_mu1 + k_mu2); v++){
                                    std::vector<double> Dv_vec {Dfun(v, p)};
                                    double sum_kpp {0.};
                                    double E3kk_0;
                                    for(int k_pp = v; k_pp >= 0; k_pp -= 2){
                                        if(k_pp <= 4){
                                            E3kk_0 = (k_pp >= k_P)? Efunt0(k_P + (k_pp*(k_pp+1))/2, p_3, PAz_3, PBz_3) : Efunt0(k_pp + (k_P*(k_P+1))/2, p_3, PBz_3, PAz_3);
                                        } else {
                                            E3kk_0 = (k_pp >= k_P)? EfunTriplet0(5*k_pp + k_P - 25, p_3, PAz_3, PBz_3) : EfunTriplet0(5*k_P + k_pp - 25, p_3, PBz_3, PAz_3);
                                        }
                                        sum_kpp += Dv_vec[k_pp/2]*E3kk_0;
                                    }
                                    sum_v += Ekk_vec(v)*sum_kpp;
                                }

                                overlap3_g_pre1 += g_mu1*g_mu2*g_P*sum_t*sum_u*sum_v;

                            }
                        }
                    }
                    overlap3_g_pre1 *= FAC3_SCF_[muind1][gaussC_mu1]*FAC3_SCF_[muind2][gaussC_mu2]*FAC3_AUX_[Pind][gaussC_P]*std::pow(p_3,-1.5)*std::exp(-(exponent_mu1*exponent_mu2*norm_mu1mu2/p) -(p*exponent_P*norm_PP_3/p_3));
                    overlap3_g_pre0 += overlap3_g_pre1;
                }
            }
        }

        overlap3_g_pre0 *= FAC12_AUX_[Pind]*FAC12_SCF_[muind1]*FAC12_SCF_[muind2]*PIpow;
        if(std::abs(overlap3_g_pre0) > etol){
            std::array<double,6> conv_vec = {overlap3_g_pre0,(double)Pind,(double)muind1,(double)muind2,(double)Rind1,(double)Rind2};
            overlap3Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

MPI_Barrier (MPI_COMM_WORLD);
// Store the matrices and the list of direct lattice vectors
uint64_t n_entries_thisproc = overlap3Matrices.size();
uint64_t n_entries_total;
MPI_Reduce(&n_entries_thisproc, &n_entries_total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

std::string o3Cstr = IntFiles_Dir + "o3Mat_" + intName + ".o3c";
if(procMPI_rank == 0){
    std::ofstream output_file0(o3Cstr, std::ios::trunc);
    output_file0.close();
    std::ofstream output_file(o3Cstr, std::ios::app);
    output_file << "3-CENTER OVERLAP INTEGRALS" << std::endl;
    output_file << "Requested nR: " << nR2 << ". Computed nR: " << nR2_star << std::endl;
    output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries_total/total_elem)*100 << " %" << std::endl;
    output_file << "Entry, P, mu, mu', R, R'" << std::endl;
    output_file << n_entries_total << std::endl;
    output_file << dimMat_AUX << " " << dimMat_SCF << std::endl;
    output_file.close();
}
MPI_Barrier (MPI_COMM_WORLD);
for(int r = 0; r < procMPI_size; r++){
    if(procMPI_rank == r){
        std::ofstream output_file(o3Cstr, std::ios::app);
        output_file.precision(12);
        output_file << std::scientific;
        for(uint64_t ent = 0; ent < n_entries_thisproc; ent++){
            output_file << overlap3Matrices[ent][0] << "  " << static_cast<uint32_t>(overlap3Matrices[ent][1]) << " " << 
                static_cast<uint32_t>(overlap3Matrices[ent][2]) << " " << static_cast<uint32_t>(overlap3Matrices[ent][3]) << " " << 
                    static_cast<uint32_t>(overlap3Matrices[ent][4]) << " " << static_cast<uint32_t>(overlap3Matrices[ent][5]) << std::endl;
        }
        output_file.close();
    }
    MPI_Barrier (MPI_COMM_WORLD);
}
if(procMPI_rank == 0){
    combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".o3c",arma::arma_ascii);
    std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
    std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "o3Mat_" + intName + ".o3c" << 
        ", and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".o3c" << std::endl;
}

}

}