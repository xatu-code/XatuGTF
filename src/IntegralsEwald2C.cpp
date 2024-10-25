#include "xatu/IntegralsEwald2C.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param scalei_supercell Vector where each component is the scaling factor for the corresponding original (unit cell) Bravais basis vectors Ri 
 *        to form the supercell. 
 * @param nR Minimum number of external supercell lattice vectors (i.e. repetitions of the supercell) to be included in the Ewald direct lattice sum.
 * @param nG Minimum number of reciprocal supercell vectors (i.e. reciprocal vectors for the lattice defined by the supercell) to be included in the Ewald reciprocal lattice sum.
 * @param intName Name of the file where the 2-center Ewald matrices will be stored as a vector (E2Mat_intName.E2c).
 */
IntegralsEwald2C::IntegralsEwald2C(const IntegralsBase& IntBase, const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const std::string& intName) 
    : IntegralsBase{IntBase} {

    setgamma0(scalei_supercell);
    Ewald2Cfun(tol, scalei_supercell, nR, nG, intName);

}

/**
 * Method to set the gamma0 attribute depending on lattice dimensionality. gamma0 is in atomic units for length^-2.
 * @param scalei_supercell Same as in the constructor, here it is used to convert the volume/are of the unit cell to that of the supercell.
 * @return void.
 */
void IntegralsEwald2C::setgamma0(const std::vector<int32_t>& scalei_supercell){

    double gamma0;
    if(ndim == 2){
        gamma0 = 2.4/std::sqrt(unitCellVolume*scalei_supercell[0]*scalei_supercell[1]);
    } 
    else if(ndim == 3){
        gamma0 = 2.8/std::cbrt(unitCellVolume*scalei_supercell[0]*scalei_supercell[1]*scalei_supercell[2]);
    }
    else{  // gamma0 irrelevant in this case, since Ewald is only used in 2D and 3D
        gamma0 = 0;
    }
    this->gamma0_ = std::pow(gamma0/ANG2AU, 2);
    this->gamma0_sqrt_ = std::sqrt(gamma0_);

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
 * @param intName Name of the file where the 2-center Ewald matrices will be stored as a vector (E2Mat_intName.E2c).
 * @return void. Matrices and the corresponding list of (inner) supercell lattice vectors are stored instead.
 */
void IntegralsEwald2C::Ewald2Cfun(const int tol, const std::vector<int32_t>& scalei_supercell, const uint32_t nR, const uint32_t nG, const std::string& intName){

#pragma omp declare reduction (merge : std::vector<std::array<double,4>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

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
const double PIpow2 = std::pow(PI,2.5) * 2.;
arma::mat combs_inner;
arma::mat RlistAU_inner   = ANG2AU*generateRlist_fixed(scalei_supercell, combs_inner, "Ewald2C"); // lattice vectors within the supercell 
uint32_t nR_inner         = RlistAU_inner.n_cols;
this->RlistAU_outer_      = ANG2AU*generateRlist_supercell(nR, scalei_supercell);                 // lattice vectors for supercell repetition
this->nR_outer_           = RlistAU_outer.n_cols;
this->GlistAU_half_       = (1./ANG2AU)*generateGlist_supercell_half(nG, scalei_supercell);       // reciprocal vectors with (reduced) supercell BZ
this->nG_                 = GlistAU_half.n_cols;
this->GlistAU_half_norms_ = arma::sqrt(arma::sum(GlistAU_half % GlistAU_half));

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat_AUX*(dimMat_AUX + 1))/2;
uint64_t total_elem = nelem_triang*nR_inner;
std::cout << "Computing " << nR_inner << " " << dimMat_AUX << "x" << dimMat_AUX << " 2-center Ewald matrices in the AUX basis..." << std::flush;

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> Ewald2Matrices;
    Ewald2Matrices.reserve(total_elem);

    #pragma omp parallel for schedule(static,1) reduction(merge: Ewald2Matrices)  
    for(uint64_t s = 0; s < total_elem; s++){ //Spans the lower triangle of all the nR_inner matrices <P,0|A|P',R>
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
                                                double Ewald2_pre2direct {0.};
                                                if( mu > gamma0_ ){
                                                    Ewald2_pre2direct = Ewald2Cdirect(coords_braket, t_tot, u_tot, v_tot, mu);
                                                }

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

// Store the matrices and the list of direct lattice vectors
uint64_t n_entries = Ewald2Matrices.size();
std::ofstream output_file(IntFiles_Dir + "E2Mat_" + intName + ".E2c");
output_file << "2-CENTER EWALD INTEGRALS (" << ndim << "D)" << std::endl;
output_file << "Supercell (inner) scaling: ";
for(int ni = 0; ni < ndim; ni++){
    output_file << scalei_supercell[ni] << " "; 
}
output_file << std::endl;
output_file << "Supercell (outer) sums: nR = " << nR_outer_ << ", nG = " << nG_ << std::endl; 
output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries/total_elem)*100 << " %" << std::endl;
output_file << "Entry, mu, mu', R" << std::endl;
output_file << n_entries << std::endl;
output_file << dimMat_AUX << std::endl;
output_file.precision(12);
output_file << std::scientific;
for(uint64_t ent = 0; ent < n_entries; ent++){
    output_file << Ewald2Matrices[ent][0] << "  " << static_cast<uint32_t>(Ewald2Matrices[ent][1]) << " " << 
    static_cast<uint32_t>(Ewald2Matrices[ent][2]) << " " << static_cast<uint32_t>(Ewald2Matrices[ent][3]) << std::endl;
}
combs_inner.save(IntFiles_Dir + "RlistFrac_" + intName + ".E2c", arma::arma_ascii);

std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "E2Mat_" + intName + ".E2c" << 
    " , and list of Bravais vectors (within supercell) in " << IntFiles_Dir + "RlistFrac_" + intName + ".E2c" << std::endl;

}

/**
 * Method to compute the direct lattice contribution to the Ewald integrals.
 * @param coords_braket Vector of coordinates of Hermite Gaussian in bra minus coordinates of Hermite Gaussian in ket, in atomic units.
 * @param t_tot Order of the Hermite Coulomb integral corresponding to the x coordinate
 * @param u_tot Order of the Hermite Coulomb integral corresponding to the y coordinate
 * @param v_tot Order of the Hermite Coulomb integral corresponding to the z coordinate
 * @param mu Reduced exponent associated to the product of Hermite Gaussians in bra and ket.
 * @return double \sum_{R}[sqrt(mu)*R_{t+t',u+u',v+v'}(mu,A-B-R) - sqrt(gamma0)*R_{t+t',u+u',v+v'}(gamma0,A-B-R)].
 */
double IntegralsEwald2C::Ewald2Cdirect(const arma::colvec& coords_braket, const int t_tot, const int u_tot, const int v_tot, const double mu){

    double Hermit_cum1 {0.};
    double Hermit_cum2 {0.};
    if(t_tot >= u_tot){
        if(u_tot >= v_tot){ // (t,u,v)
            for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
                arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
                Hermit_cum1 += HermiteCoulomb(t_tot, u_tot, v_tot, mu,      coords_braket_Rn(0), coords_braket_Rn(1), coords_braket_Rn(2));
                Hermit_cum2 += HermiteCoulomb(t_tot, u_tot, v_tot, gamma0_, coords_braket_Rn(0), coords_braket_Rn(1), coords_braket_Rn(2));
            }
        }
        else if(t_tot >= v_tot){ // (t,v,u)
            for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
                arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
                Hermit_cum1 += HermiteCoulomb(t_tot, v_tot, u_tot, mu,      coords_braket_Rn(0), coords_braket_Rn(2), coords_braket_Rn(1));
                Hermit_cum2 += HermiteCoulomb(t_tot, v_tot, u_tot, gamma0_, coords_braket_Rn(0), coords_braket_Rn(2), coords_braket_Rn(1));
            }
        }
        else{ // (v,t,u)
            for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
                arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
                Hermit_cum1 += HermiteCoulomb(v_tot, t_tot, u_tot, mu,      coords_braket_Rn(2), coords_braket_Rn(0), coords_braket_Rn(1));
                Hermit_cum2 += HermiteCoulomb(v_tot, t_tot, u_tot, gamma0_, coords_braket_Rn(2), coords_braket_Rn(0), coords_braket_Rn(1));
            }
        }
    } 
    else if(u_tot >= v_tot){ 
        if(t_tot >= v_tot){ // (u,t,v)
            for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
                arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
                Hermit_cum1 += HermiteCoulomb(u_tot, t_tot, v_tot, mu,      coords_braket_Rn(1), coords_braket_Rn(0), coords_braket_Rn(2));
                Hermit_cum2 += HermiteCoulomb(u_tot, t_tot, v_tot, gamma0_, coords_braket_Rn(1), coords_braket_Rn(0), coords_braket_Rn(2));
            }
        }
        else{ // (u,v,t)
            for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
                arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
                Hermit_cum1 += HermiteCoulomb(u_tot, v_tot, t_tot, mu,      coords_braket_Rn(1), coords_braket_Rn(2), coords_braket_Rn(0));
                Hermit_cum2 += HermiteCoulomb(u_tot, v_tot, t_tot, gamma0_, coords_braket_Rn(1), coords_braket_Rn(2), coords_braket_Rn(0));
            }
        }
    }
    else{ // (v,u,t)
        for(uint32_t Rind = 0; Rind < nR_outer_; Rind++){
            arma::colvec coords_braket_Rn {coords_braket - RlistAU_outer_.col(Rind)};
            Hermit_cum1 += HermiteCoulomb(v_tot, u_tot, t_tot, mu,      coords_braket_Rn(2), coords_braket_Rn(1), coords_braket_Rn(0));
            Hermit_cum2 += HermiteCoulomb(v_tot, u_tot, t_tot, gamma0_, coords_braket_Rn(2), coords_braket_Rn(1), coords_braket_Rn(0));
        }
    }
                                                        
    return ( std::sqrt(mu)*Hermit_cum1 - gamma0_sqrt_*Hermit_cum2 );
    
}

/**
 * Method to compute the reciprocal lattice contribution to the Ewald integrals, which depends on the dimensionality of the system (2D or 3D).
 * @param coords_braket Vector of coordinates of Hermite Gaussian in bra minus coordinates of Hermite Gaussian in ket, in atomic units.
 * @param t_tot Order of the Hermite Coulomb integral corresponding to the x coordinate.
 * @param u_tot Order of the Hermite Coulomb integral corresponding to the y coordinate.
 * @param v_tot Order of the Hermite Coulomb integral corresponding to the z coordinate.
 * @param gamma_fac The minimum of mu and gamma0.
 * @return double [G=0 term + sum_{G} (recip_term)] in 2D, or  sum_{G} (recip_term) in 3D.
 */
double IntegralsEwald2C::Ewald2Creciprocal(const arma::colvec& coords_braket, const int t_tot, const int u_tot, const int v_tot, const double gamma_fac){

    double recip_G0 {0.};
    double recip_cum {0.};
    if(ndim == 2){

        int sum_indtot_xy {t_tot + u_tot};
        double gamma_fac_sqrt {std::sqrt(gamma_fac)};
        double bz {gamma_fac_sqrt * coords_braket(2)};
        double b_pow {std::pow(2*gamma_fac_sqrt, v_tot)};

        // G = 0 contribution
        if(sum_indtot_xy == 0){
            recip_G0 += derivative_G02Dz(v_tot, bz, gamma_fac_sqrt);
        }

        // Sum over G /= 0 (excluding opposites)
        double parity_vtot {((v_tot % 2) == 0)? 1. : -1.};
        for(uint32_t Gind = 0; Gind < nG_; Gind++){
            arma::colvec Gn {GlistAU_half_.col(Gind)};
            double norm_Gn {GlistAU_half_norms_(Gind)};
            double a {0.5*norm_Gn / gamma_fac_sqrt};
            double cosGn_fac {derivative_cos( sum_indtot_xy, arma::dot(Gn, coords_braket) ) * ( std::pow(Gn(0),t_tot) * std::pow(Gn(1),u_tot) )};
            recip_cum += (cosGn_fac / norm_Gn) * (derivative_Gfinite2Dz(v_tot, a, bz, b_pow) + parity_vtot*derivative_Gfinite2Dz(v_tot, a, -bz, b_pow));
        }

    } 
    else if(ndim == 3){

        // Sum over G /= 0 (excluding opposites)
        int sum_indtot {t_tot + u_tot + v_tot};
        for(uint32_t Gind = 0; Gind < nG_; Gind++){
            arma::colvec Gn {GlistAU_half_.col(Gind)};
            double normsq_Gn {std::pow(GlistAU_half_norms_(Gind), 2)};
            double cosGn_fac {derivative_cos( sum_indtot, arma::dot(Gn, coords_braket) ) * ( std::pow(Gn(0),t_tot) * std::pow(Gn(1),u_tot) * std::pow(Gn(2),v_tot) )};
            recip_cum += (cosGn_fac / normsq_Gn) * std::exp( -0.25*normsq_Gn/gamma_fac );
        }
        recip_cum *= 4.;

    }
    else{
        throw std::invalid_argument("ERROR Ewald2Creciprocal: the Ewald potential can only be used in 2D or 3D systems");
    }
    
    return ( recip_cum - recip_G0 );
    
}

/**
 * Method to evaluate the n-th derivative of the cosine function at arg.
 * @param n Order of the derivative, possibly 0 (in which case the derivative is the identity operator).
 * @param arg Argument in which the derivative is evaluated.
 * @return double d^n/dx^n cos(x) | x = arg.
 */
double IntegralsEwald2C::derivative_cos(const int n, const double arg){

    switch(n % 4)
    {
    case 0: {
        return std::cos(arg);
    }
    case 1: {
        return -std::sin(arg);
    }
    case 2: {
        return -std::cos(arg);
    }
    case 3: {
        return std::sin(arg);
    }
    default: 
        throw std::invalid_argument("IntegralsEwald2C::derivative_cos error: this statement should never be reached");
    }

}

/**
 * Method to evaluate the n-th derivative with respect to (r_1z - r_2z) of the G = 0 term in the 2D Ewald potential.
 * Global minus sign is NOT included.
 * @param v_tot Order of the derivative, possibly 0 (in which case the derivative is the identity operator).
 * @param bz Argument of the erf function, and also the square root of the argument of the exponential: sqrt(min(mu,gamma0))*(r_1z - r_2z).
 * @param b sqrt(min(mu,gamma0)).
 * @return double The aforementioned derivative evaluated at the appropriate argument.
 */
double IntegralsEwald2C::derivative_G02Dz(const int v_tot, const double bz, const double b){

    switch(v_tot)
    {
    case 0: {
        return (bz/b)*std::erf(bz) + (PISQRT_INV/b)*std::exp(-bz*bz);
    }
    case 1: {
        return std::erf(bz);
    }
    case 2: {
        return PISQRT_INV*2*b*std::exp(-bz*bz);
    }
    case 3: {
        return -PISQRT_INV*4*b*b*bz*std::exp(-bz*bz);
    }
    case 4: {
        return PISQRT_INV*4*b*b*b*(2*bz*bz - 1)*std::exp(-bz*bz);
    }
    case 5: {
        return PISQRT_INV*8*b*b*b*b*bz*(3 - 2*bz*bz)*std::exp(-bz*bz);
    }
    case 6: {
        double bz2 {bz*bz};
        return PISQRT_INV*8*std::pow(b,5)*(3 - 12*bz2 + 4*bz2*bz2)*std::exp(-bz2);
    }
    case 7: {
        double bz2 {bz*bz};
        return -PISQRT_INV*16*std::pow(b,6)*bz*(15 - 20*bz2 + 4*bz2*bz2)*std::exp(-bz2);
    }
    case 8: {
        double bz2 {bz*bz};
        return PISQRT_INV*16*std::pow(b,7)*(8*bz2*bz2*bz2 - 60*bz2*bz2 + 90*bz2 - 15)*std::exp(-bz2);
    }
    default: {
        throw std::invalid_argument("IntegralsEwald2C::derivative_G02Dz error: a derivative higher than 8-th with respect to r_z is being evaluated");
    }
    }

}

/**
 * Method to evaluate the n-th derivative with respect to (r_1z - r_2z) of the term in the 2D Ewald potential that is summed over G.
 * It only evaluates one of the two terms in: f(r_1z - r_2z) + f(-(r_1z - r_2z)).
 * @param v_tot Order of the derivative, possibly 0 (in which case the derivative is the identity operator).
 * @param a Constant term in the erfc function (with respect to the derivative): norm(Gn)/(2*sqrt(min(mu,gamma0))).
 * @param bz Term in the erfc to be differentiated: sqrt(min(mu,gamma0))*(r_1z - r_2z).
 * @param b_pow (2*sqrt(min(mu,gamma0)))^v_tot.
 * @return double The aforementioned derivative evaluated at the appropriate argument.
 */
double IntegralsEwald2C::derivative_Gfinite2Dz(const int v_tot, const double a, const double bz, const double b_pow){

    switch(v_tot)
    {
    case 0: {
        return std::exp(2*a*bz)*std::erfc(a + bz);
    }
    case 1: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        return b_pow*std::exp(2*a*bz)*(a*std::erfc(asumbz) - exp2pi_fac);
    }
    case 2: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        return b_pow*std::exp(2*a*bz)*(a*a*std::erfc(a + bz) - (a - bz)*exp2pi_fac);
    }
    case 3: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        return b_pow*std::exp(2*a*bz)*(a*a*a*std::erfc(asumbz) - (a*a + bz*(bz - a) - 0.5)*exp2pi_fac);
    }
    case 4: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        double a2 {a*a};
        return b_pow*std::exp(2*a*bz)*(a2*a2*std::erfc(asumbz) + ( a2*(bz - a) + 0.5*a + bz*(bz*(bz - a) - 1.5) )*exp2pi_fac);
    }
    case 5: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        double a2 {a*a};
        return b_pow*std::exp(2*a*bz)*(a2*a2*a*std::erfc(asumbz) + ( -a2*(a2 - 0.5) + a*bz*(a2 + bz*(bz - a) - 1.5) - bz*bz*(bz*bz - 3) - 0.75 )*exp2pi_fac);
    }
    case 6: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        double a2 {a*a};
        double a3 {a2*a};
        double bz2 {bz*bz};
        double aux_fac {-a3*(a2 - 0.5) - a2*bz2*(a - bz) + a*bz*(a3 - 1.5*a + bz*(3 - bz2)) + bz2*bz*(bz2 - 5) - 0.75*a + 3.75*bz};
        return b_pow*std::exp(2*a*bz)*(a3*a3*std::erfc(asumbz) + aux_fac*exp2pi_fac);
    }
    case 7: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        double a2 {a*a};
        double a3 {a2*a};
        double bz2 {bz*bz};
        double aux_fac {-a2*a2*(a2 - 0.5) - a2*bz2*(a2 + bz2 - a*bz - 3) + a*bz*(a2*a2 - 1.5*a2 + bz2*(bz2 - 5) + 3.75) - bz2*(bz2*(bz2 - 7.5) + 11.25) - 0.75*a2 + 1.875};
        return b_pow*std::exp(2*a*bz)*(a3*a3*a*std::erfc(asumbz) + aux_fac*exp2pi_fac);
    }
    case 8: {
        double asumbz = a + bz;
        double exp2pi_fac {PISQRT_INV*std::exp( -asumbz*asumbz )};
        double a2 {a*a};
        double a3 {a2*a};
        double a4 {a2*a2};
        double bz2 {bz*bz};
        double aux_fac {-a4*a*(a2 - 0.5) - 0.75*a3 + 1.875*a + a3*bz2*bz*(a - bz) - a2*bz2*(a3 - 3*a - bz*(bz2 - 5)) + a*bz*(a*(a4 - 1.5*a2 + 3.75) - bz*(bz2*(bz2 - 7.5) + 11.25))};
        aux_fac += bz*(bz2*(bz2*(bz2 - 10.5) + 26.25) - 13.125);
        return b_pow*std::exp(2*a*bz)*(a4*a4*std::erfc(asumbz) + aux_fac*exp2pi_fac);
    }
    default: {
        throw std::invalid_argument("IntegralsEwald2C::derivative_Gfinite2Dz error: a derivative higher than 8-th with respect to r_z is being evaluated");
    }
    }

}

}