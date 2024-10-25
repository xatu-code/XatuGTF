#include "xatu/IntegralsDipole.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase object.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only triplets of entries (x,y,z) where at least one element is > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the dipole integrals will be computed.
 * @param intName Name of the file where the dipole matrices will be stored as a vector (dipoleMat_intName.dip).
 */
IntegralsDipole::IntegralsDipole(const IntegralsBase& IntBase, const int tol, const uint32_t nR, const std::string& intName) : IntegralsBase{IntBase} {

    dipolefun(tol, nR, intName);

}

/**
 * Method to compute the dipole matrices <mu,0|r|mu',R>, r = x,y,z, in the SCF basis set for the first nR Bravais vectors R. 
 * These first nR (at least, until the star of vectors is completed) are generated with Lattice::generateRlist. Each triplet of entries
 * (x,y,z) where at least one element is above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the 
 * corresponding indices: value_x,value_y,value_z,mu,mu',R; in that order. The vector is saved in the dipoleMat_intName.dip file in ATOMIC
 * UNITS, and the list of Bravais vectors in fractional coordinates is saved in the RlistFrac_intName.dip file. The whole R-matrices are 
 * stored (both triangles). 
 * @param tol Threshold tolerance for the integrals: only triplets of entries (x,y,z) where at least one element is > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the dipole integrals will be computed.
 * @param intName Name of the file where the dipole matrices will be stored as a vector (dipoleMat_intName.dip).
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsDipole::dipolefun(const int tol, const uint32_t nR, const std::string& intName){

#pragma omp declare reduction (merge : std::vector<std::array<double,6>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

const double PIpow = std::pow(PI,1.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR, combs, "Dipole");  //convert Bravais vectors from Angstrom to atomic units
uint32_t nR_star = RlistAU.n_cols;
std::map<uint32_t,uint32_t> RlistOpposites = generateRlistOpposite(RlistAU);

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat_SCF*(dimMat_SCF + 1))/2;
uint64_t total_elem = nelem_triang*nR_star;

std::cout << "Computing " << nR_star << " " << dimMat_SCF << "x" << dimMat_SCF << " dipole matrices (x,y,z) in the SCF basis..." << std::flush;

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,6>> dipoleMatrices;
    dipoleMatrices.reserve(dimMat_SCF*dimMat_SCF*nR_star); // includes the upper triangle, reconstructed from the lower

    #pragma omp parallel for schedule(static,1) reduction(merge: dipoleMatrices)  
    for(uint64_t s = 0; s < total_elem; s++){ //Spans the lower triangle of all the nR_star matrices <mu,0|r|mu',R>
        uint64_t sind {s % nelem_triang};     //Index for the corresponding entry in the dipole matrices, irrespective of the specific R
        uint32_t Rind {static_cast<uint32_t>(s / nelem_triang)};         //Position in RlistAU (e.g. 0 for R=0) of the corresponding Bravais vector 
        uint64_t r = nelem_triang - (sind + 1);
        uint64_t l = (std::sqrt(8*r + 1) - 1)/2;
        uint32_t orb_bra = (sind + dimMat_SCF + (l*(l+1))/2 ) - nelem_triang;  //Orbital number (<dimMat) of the bra corresponding to the index s 
        uint32_t orb_ket = dimMat_SCF - (l + 1);                               //Orbital number (<dimMat) of the ket corresponding to the index s. orb_ket <= orb_bra (lower triangle)
        arma::colvec R {RlistAU.col(Rind)};  //Bravais vector (a.u.) corresponding to the "s" matrix element
        uint32_t RindOpp    {RlistOpposites.at(Rind)};  //Position in RlistAU (i.e. 0 for R=0) of the opposite of the corresponding Bravais vector 
        
        int L_bra  {orbitals_info_int_SCF[orb_bra][2]};
        int m_bra  {orbitals_info_int_SCF[orb_bra][3]};
        int nG_bra {orbitals_info_int_SCF[orb_bra][4]};
        arma::colvec coords_bra {orbitals_info_real_SCF[orb_bra][0], orbitals_info_real_SCF[orb_bra][1], orbitals_info_real_SCF[orb_bra][2]};  //Position (a.u.) of bra atom
        std::vector<int> g_coefs_bra   {g_coefs_.at( L_bra*(L_bra + 1) + m_bra )};

        int L_ket  {orbitals_info_int_SCF[orb_ket][2]};
        int m_ket  {orbitals_info_int_SCF[orb_ket][3]};
        int nG_ket {orbitals_info_int_SCF[orb_ket][4]};
        arma::colvec coords_ket {R + arma::colvec{orbitals_info_real_SCF[orb_ket][0], orbitals_info_real_SCF[orb_ket][1], orbitals_info_real_SCF[orb_ket][2]} };  //Position (a.u.) of ket atom
        std::vector<int> g_coefs_ket   {g_coefs_.at( L_ket*(L_ket + 1) + m_ket )};

        double norm_braket {arma::dot(coords_bra - coords_ket, coords_bra - coords_ket)};
        double FAC12_braket = FAC12_SCF[orb_bra]*FAC12_SCF[orb_ket];

        double dipoleX_pre0 {0.};  // x component
        double dipoleY_pre0 {0.};  // y component
        double dipoleZ_pre0 {0.};  // z component
        double overlap_pre0 {0.};
        for(int gaussC_bra = 0; gaussC_bra < nG_bra; gaussC_bra++){ //Iterate over the contracted Gaussians in the bra orbital
            double exponent_bra {orbitals_info_real_SCF[orb_bra][2*gaussC_bra + 3]};
            //double d_bra {orbitals_info_real[orb_bra][2*gaussC_bra + 4]};

            for(int gaussC_ket = 0; gaussC_ket < nG_ket; gaussC_ket++){ //Iterate over the contracted Gaussians in the ket orbital
                double exponent_ket {orbitals_info_real_SCF[orb_ket][2*gaussC_ket + 3]};
                //double d_ket {orbitals_info_real[orb_ket][2*gaussC_ket + 4]};

                double p {exponent_bra + exponent_ket};  //Exponent coefficient of the Hermite Gaussian
                arma::colvec P {(exponent_bra*coords_bra + exponent_ket*coords_ket)/p};  //Center of the Hermite Gaussian
                double PAx {P(0) - coords_bra(0)}; 
                double PAy {P(1) - coords_bra(1)}; 
                double PAz {P(2) - coords_bra(2)}; 
                double PBx {P(0) - coords_ket(0)}; 
                double PBy {P(1) - coords_ket(1)}; 
                double PBz {P(2) - coords_ket(2)}; 

                double dipoleX_pre1 {0.};
                double dipoleY_pre1 {0.};
                double dipoleZ_pre1 {0.};
                double overlap_pre1 {0.};
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
                        int Ei_ket {(i_ket*(i_ket + 1))/2};
                        int Ej_ket {(j_ket*(j_ket + 1))/2};
                        int Ek_ket {(k_ket*(k_ket + 1))/2};

                        // X contribution
                        double Eii0 {(i_bra >= i_ket)? Efunt0(i_ket + Ei_bra, p, PAx, PBx) : Efunt0(i_bra + Ei_ket, p, PBx, PAx)};
                        arma::colvec Eii_vec {(i_bra >= i_ket)? Efun(i_ket + Ei_bra, p, PAx, PBx) : Efun(i_bra + Ei_ket, p, PBx, PAx)};
                        double sum_t {0.};
                        for(int t = 0; t <= (i_bra + i_ket); t++){
                            std::vector<double> Dt_vec {Dfun(t, p)};
                            double sum_l {0.};
                            double El1_0;
                            for(int l = t; l >= 0; l -= 2){
                                if(l <= 4){
                                    El1_0 = (l >= 1)? Efunt0(1 + (l*(l+1))/2, p, 0., P(0)) : Efunt0(l + 1, p, P(0), 0.);
                                } else {
                                    El1_0 = (l >= 1)? EfunTriplet0(5*l -24, p, 0., P(0)) : EfunTriplet0(l - 20, p, P(0), 0.);
                                }
                                sum_l += Dt_vec[l/2]*El1_0;
                            }
                            sum_t += Eii_vec(t)*sum_l;
                        }  
                        // Y contribution
                        double Ejj0 {(j_bra >= j_ket)? Efunt0(j_ket + Ej_bra, p, PAy, PBy) : Efunt0(j_bra + Ej_ket, p, PBy, PAy)};
                        arma::colvec Ejj_vec {(j_bra >= j_ket)? Efun(j_ket + Ej_bra, p, PAy, PBy) : Efun(j_bra + Ej_ket, p, PBy, PAy)};
                        double sum_u {0.};
                        for(int u = 0; u <= (j_bra + j_ket); u++){
                            std::vector<double> Dt_vec {Dfun(u, p)};
                            double sum_m {0.};
                            double Em1_0;
                            for(int m = u; m >= 0; m -= 2){
                                if(m <= 4){
                                    Em1_0 = (m >= 1)? Efunt0(1 + (m*(m+1))/2, p, 0., P(1)) : Efunt0(m + 1, p, P(1), 0.);
                                } else {
                                    Em1_0 = (m >= 1)? EfunTriplet0(5*m -24, p, 0., P(1)) : EfunTriplet0(m - 20, p, P(1), 0.);
                                }
                                sum_m += Dt_vec[m/2]*Em1_0;
                            }
                            sum_u += Ejj_vec(u)*sum_m;
                        } 
                        // Z contribution
                        double Ekk0 {(k_bra >= k_ket)? Efunt0(k_ket + Ek_bra, p, PAz, PBz) : Efunt0(k_bra + Ek_ket, p, PBz, PAz)};
                        arma::colvec Ekk_vec {(k_bra >= k_ket)? Efun(k_ket + Ek_bra, p, PAz, PBz) : Efun(k_bra + Ek_ket, p, PBz, PAz)};
                        double sum_v {0.};
                        for(int v = 0; v <= (k_bra + k_ket); v++){
                            std::vector<double> Dt_vec {Dfun(v, p)};
                            double sum_n {0.};
                            double En1_0;
                            for(int n = v; n >= 0; n -= 2){
                                if(n <= 4){
                                    En1_0 = (n >= 1)? Efunt0(1 + (n*(n+1))/2, p, 0., P(2)) : Efunt0(n + 1, p, P(2), 0.);
                                } else {
                                    En1_0 = (n >= 1)? EfunTriplet0(5*n -24, p, 0., P(2)) : EfunTriplet0(n - 20, p, P(2), 0.);
                                }
                                sum_n += Dt_vec[n/2]*En1_0;
                            }
                            sum_v += Ekk_vec(v)*sum_n;
                        }

                        dipoleX_pre1 += g_bra*g_ket*sum_t*Ejj0*Ekk0;
                        dipoleY_pre1 += g_bra*g_ket*Eii0*sum_u*Ekk0;
                        dipoleZ_pre1 += g_bra*g_ket*Eii0*Ejj0*sum_v;
                        overlap_pre1 += g_bra*g_ket*Eii0*Ejj0*Ekk0;

                    }
                }
                double dipole_pre1_fac {FAC3_SCF[orb_bra][gaussC_bra]*FAC3_SCF[orb_ket][gaussC_ket]*std::pow(p,-1.5)*std::exp(-exponent_bra*exponent_ket*norm_braket/p)};
                dipoleX_pre1 *= dipole_pre1_fac;
                dipoleX_pre0 += dipoleX_pre1;
                dipoleY_pre1 *= dipole_pre1_fac;
                dipoleY_pre0 += dipoleY_pre1;
                dipoleZ_pre1 *= dipole_pre1_fac;
                dipoleZ_pre0 += dipoleZ_pre1;
                overlap_pre1 *= dipole_pre1_fac;
                overlap_pre0 += overlap_pre1;
            }
        }
        double dipole_pre0_fac {FAC12_braket*PIpow};
        dipoleX_pre0 *= dipole_pre0_fac;
        dipoleY_pre0 *= dipole_pre0_fac;
        dipoleZ_pre0 *= dipole_pre0_fac;
        overlap_pre0 *= dipole_pre0_fac;
        if(std::abs(dipoleX_pre0) > etol || std::abs(dipoleY_pre0) > etol || std::abs(dipoleZ_pre0) > etol){
            std::array<double,6> conv_vec = {dipoleX_pre0, dipoleY_pre0, dipoleZ_pre0, (double)orb_bra, (double)orb_ket, (double)Rind};
            dipoleMatrices.push_back(conv_vec);
        }
        if(orb_bra > orb_ket){  // complete the upper triangle
            double dipoleX_pre0_upper {dipoleX_pre0 - R(0)*overlap_pre0};
            double dipoleY_pre0_upper {dipoleY_pre0 - R(1)*overlap_pre0};
            double dipoleZ_pre0_upper {dipoleZ_pre0 - R(2)*overlap_pre0};
            if(std::abs(dipoleX_pre0_upper) > etol || std::abs(dipoleY_pre0_upper) > etol || std::abs(dipoleZ_pre0_upper) > etol){
                std::array<double,6> conv_vec_opp = {dipoleX_pre0_upper, dipoleY_pre0_upper, dipoleZ_pre0_upper, (double)orb_ket, (double)orb_bra, (double)RindOpp};
                dipoleMatrices.push_back(conv_vec_opp);
            }
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

// Store the matrices and the list of direct lattice vectors
uint64_t n_entries = dipoleMatrices.size();   
std::ofstream output_file(IntFiles_Dir + "dipoleMat_" + intName + ".dip");
output_file << "DIPOLE INTEGRALS" << std::endl;
output_file << "Requested nR: " << nR << ". Computed nR: " << nR_star << std::endl;
output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries/(dimMat_SCF*dimMat_SCF*nR_star))*100 << " %" << std::endl;
output_file << "Entry X, Entry Y, Entry Z, mu, mu', R" << std::endl;
output_file << n_entries << std::endl;
output_file << dimMat_SCF << std::endl;
output_file.precision(12);
output_file << std::scientific;
for(uint64_t ent = 0; ent < n_entries; ent++){
    output_file << dipoleMatrices[ent][0] << " " << dipoleMatrices[ent][1] << " " << dipoleMatrices[ent][2] << "  " << 
        static_cast<uint32_t>(dipoleMatrices[ent][3]) << " " << static_cast<uint32_t>(dipoleMatrices[ent][4]) << " " << 
            static_cast<uint32_t>(dipoleMatrices[ent][5]) << std::endl;
}
combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".dip", arma::arma_ascii);

std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "dipoleMat_" + intName + ".dip" << 
    " , and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".dip" << std::endl;

}

}