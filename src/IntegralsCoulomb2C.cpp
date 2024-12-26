#include "xatu/IntegralsCoulomb2C.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nRi Vector with the number of values that the corresponding fractional coordinates (of the direct lattice vectors for which 
 * the 2-center Coulomb integrals will be computed) span. The fractional coordinates are always centered at 0.
 * @param intName Name of the file where the 2-center Coulomb matrices will be stored as a vector (C2Mat_intName.C2c).
 */
IntegralsCoulomb2C::IntegralsCoulomb2C(const IntegralsBase& IntBase, const int tol, const std::vector<int32_t>& nRi, const std::string& intName) : IntegralsBase{IntBase} {

    Coulomb2Cfun(tol, nRi, intName);

}

/**
 * Method to compute the Coulomb matrices in the auxiliary basis (<P,0|V_c|P',R>) for a set of Bravais vectors whose fractional
 * components span the number of values given in the vector nRi. This construction of R vectors is not by norm as in the other 
 * integrals due to the conditional convergence of Coulomb integrals, and it must be chosen so that nRi[n] is a multiple of nki[n].
 * Each entry above a certain tolerance (10^-tol) is stored in an entry of a vector (of arrays) along with the corresponding 
 * indices: value,mu,mu',R; in that order. The vector is saved in the C2Mat_intName.C2c file file, and the list of Bravais 
 * vectors in fractional coordinates is saved in the RlistFrac_intName.C2c file. Only the lower triangle of each R-matrix is stored; 
 * the upper triangle is given by hermiticity in the k-matrix.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nRi Vector with the number of values that the corresponding fractional coordinates (of the direct lattice vectors for which 
 * the 2-center Coulomb integrals will be computed) span. The fractional coordinates are always centered at 0.
 * @param intName Name of the file where the 2-center Coulomb matrices will be stored as a vector (C2Mat_intName.C2c).
 * @return void. Matrices and the corresponding list of lattice vectors are stored instead.
 */
void IntegralsCoulomb2C::Coulomb2Cfun(const int tol, const std::vector<int32_t>& nRi, const std::string& intName){

#pragma omp declare reduction (merge : std::vector<std::array<double,4>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

const double PIpow = std::pow(PI,2.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist_fixed(nRi, combs, "Coulomb2C");
uint32_t nR_star = RlistAU.n_cols;

double etol = std::pow(10.,-tol);
uint64_t nelem_triang = (dimMat_AUX*(dimMat_AUX + 1))/2;
uint64_t total_elem = nelem_triang*nR_star;
std::cout << "Computing " << nR_star << " " << dimMat_AUX << "x" << dimMat_AUX << " 2-center Coulomb matrices in the AUX basis..." << std::flush;

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> Coulomb2Matrices;
    Coulomb2Matrices.reserve(total_elem);

    #pragma omp parallel for schedule(static,1) reduction(merge: Coulomb2Matrices)  
    for(uint64_t s = 0; s < total_elem; s++){ //Spans the lower triangle of all the nR_star matrices <P,0|V_c|P',R>
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

        double Coulomb2_g_pre0 {0.};
        for(int gaussC_bra = 0; gaussC_bra < nG_bra; gaussC_bra++){ //Iterate over the contracted Gaussians in the bra orbital
            double exponent_bra {orbitals_info_real_AUX_[orb_bra][2*gaussC_bra + 3]};
            //double d_bra {orbitals_info_real_AUX[orb_bra][2*gaussC_bra + 4]};

            for(int gaussC_ket = 0; gaussC_ket < nG_ket; gaussC_ket++){ //Iterate over the contracted Gaussians in the ket orbital
                double exponent_ket {orbitals_info_real_AUX_[orb_ket][2*gaussC_ket + 3]};
                //double d_ket {orbitals_info_real_AUX[orb_ket][2*gaussC_ket + 4]};

                double p {exponent_bra + exponent_ket};  //Exponent coefficient of the Hermite Gaussian
                double mu {exponent_bra*exponent_ket/p};

                double Coulomb2_g_pre1 {0.};
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
                                                double Hermit;
                                                if(t_tot >= u_tot){
                                                    if(u_tot >= v_tot){ // (t,u,v)
                                                        Hermit = HermiteCoulomb(t_tot, u_tot, v_tot, mu, coords_braket(0), coords_braket(1), coords_braket(2));
                                                    }
                                                    else if(t_tot >= v_tot){ // (t,v,u)
                                                        Hermit = HermiteCoulomb(t_tot, v_tot, u_tot, mu, coords_braket(0), coords_braket(2), coords_braket(1));
                                                    }
                                                    else{ // (v,t,u)
                                                        Hermit = HermiteCoulomb(v_tot, t_tot, u_tot, mu, coords_braket(2), coords_braket(0), coords_braket(1));
                                                    }
                                                } 
                                                else if(u_tot >= v_tot){ 
                                                    if(t_tot >= v_tot){ // (u,t,v)
                                                        Hermit = HermiteCoulomb(u_tot, t_tot, v_tot, mu, coords_braket(1), coords_braket(0), coords_braket(2));
                                                    }
                                                    else{ // (u,v,t)
                                                        Hermit = HermiteCoulomb(u_tot, v_tot, t_tot, mu, coords_braket(1), coords_braket(2), coords_braket(0));
                                                    }
                                                }
                                                else{ // (v,u,t)
                                                    Hermit = HermiteCoulomb(v_tot, u_tot, t_tot, mu, coords_braket(2), coords_braket(1), coords_braket(0));
                                                }
                                                
                                                Coulomb2_g_pre1 += g_bra*g_ket*sign_ket*Eijk_bra*Ei0vec_ket(t_ket)*Ej0vec_ket(u_ket)*Ek0vec_ket(v_ket)*Hermit;

                                            }
                                        }
                                    }

                                }
                            }
                        }


                    }
                }
                Coulomb2_g_pre1 *= FAC3_AUX_[orb_bra][gaussC_bra]*FAC3_AUX_[orb_ket][gaussC_ket]*std::pow(p,-1.5)/mu;
                Coulomb2_g_pre0 += Coulomb2_g_pre1;

            }
        }
        Coulomb2_g_pre0 *= FAC12_braket*2*PIpow;
        if(std::abs(Coulomb2_g_pre0) > etol){
            std::array<double,4> conv_vec = {Coulomb2_g_pre0,(double)orb_bra,(double)orb_ket,(double)Rind};
            Coulomb2Matrices.push_back(conv_vec);
        }

    }

    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); 

// Store the matrices and the list of direct lattice vectors
uint64_t n_entries = Coulomb2Matrices.size();
std::ofstream output_file(IntFiles_Dir + "C2Mat_" + intName + ".C2c");
output_file << "2-CENTER COULOMB INTEGRALS" << std::endl;
output_file << "Requested nR: ";
for(int ni = 0; ni < ndim; ni++){
    output_file << nRi[ni] << " "; 
}
output_file << std::endl;
output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries/total_elem)*100 << " %" << std::endl;
output_file << "Entry, mu, mu', R" << std::endl;
output_file << n_entries << std::endl;
output_file << dimMat_AUX << std::endl;
output_file.precision(12);
output_file << std::scientific;
for(uint64_t ent = 0; ent < n_entries; ent++){
    output_file << Coulomb2Matrices[ent][0] << "  " << static_cast<uint32_t>(Coulomb2Matrices[ent][1]) << " " << 
    static_cast<uint32_t>(Coulomb2Matrices[ent][2]) << " " << static_cast<uint32_t>(Coulomb2Matrices[ent][3]) << std::endl;
}
combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".C2c", arma::arma_ascii);

std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "C2Mat_" + intName + ".C2c" << 
    " , and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".C2c" << std::endl;

}

/**
 * Analogous to Efun in the parent class IntegralsBase, but restricted to i'=0 and setting PA=0. These are the 
 * expansion coefficients of a single (cartesian) GTF in Hermite Gaussians with the same exponent and center.
 * @param i Index i of the cartesian Gaussian.
 * @param p Exponent of the cartesian Gaussian.
 * @return arma::colvec Vector where each entry indicates a value of t, and contains E^{i,0}_{t}.
 */
arma::colvec IntegralsCoulomb2C::Efun_single(const int i, const double p){

    switch(i)
    {
    case 0:  { 
        return arma::colvec {1.0};
    } 
    case 1:  { 
        return arma::colvec {0, 0.5/p};
    }
    case 2:  { 
        double facp = 0.5/p;
        return arma::colvec {facp,  0,  facp*facp};
    }
    case 3:  { 
        double facp = 0.5/p;
        double facp_to2 = facp*facp;
        return arma::colvec {0,  3*facp_to2,  0,  facp_to2*facp};
    }
    case 4:  { 
        double facp = 0.5/p;
        double facp_to2 = facp*facp;
        double facp_to3 = facp_to2*facp;
        return arma::colvec {3*facp_to2,  0,  6*facp_to3,  0,  facp_to3*facp};
    }
    default: {
        throw std::invalid_argument("IntegralsCoulomb2C::Efun_single error: the E^{i,0}_{t} coefficients are being evaluated for i >= 5");
    }
    }

}

/**
 * Method to compute and return the Boys function F_{n}(arg) = \int_{0}^{1}t^{2n}exp(-arg*t^2)dt. It is computed 
 * with the lower incomplete Gamma function as: F_{n}(arg) = Gamma(n+0.5)*IncGamma(n+0.5,arg)/(2*arg^(n+0.5)), see (9.8.20)-Helgaker.
 * @param n Order of the Boys function.
 * @param arg Argument of the Boys function.
 * @return double Value of the Boys function.
 */
double IntegralsCoulomb2C::Boysfun(const int n, const double arg){

    if(arg < 3.1e-10){
        return (double)1./(2*n + 1);
    }
    else if(arg > 26.5){
        return doubleFactorial(2*n-1)*std::pow(2,-n-1)*std::sqrt(PI*std::pow(arg,-2*n-1));
    }
    else{
        double nfac = n + 0.5;
        return asa239::gammad(arg, nfac)*std::tgamma(nfac)*0.5*std::pow(arg,-nfac);
    }

}

/**
 * Method to compute and return the auxiliary Hermite Coulomb integral R^{n}_{0,0,0}(p,arg)=(-2p)^n *F_{n}(arg), see (9.9.14)-Helgaker.
 * @param n Order of the auxiliary Hermite Coulomb integral.
 * @param p First argument of the auxiliary Hermite Coulomb integral.
 * @param arg Second argument of the auxiliary Hermite Coulomb integral.
 * @return double Value of the auxiliary Hermite Coulomb integral.
 */
double IntegralsCoulomb2C::Rn000(const int n, const double p, const double arg){

    return std::pow(-2*p,n)*Boysfun( n, arg );

}

/**
 * Method to compute and return the Hermite Coulomb integral R^{0}_{t,u,v}(r,(X,Y,Z)), see (9.9.9)-Helgaker.
 * Restricted to (t + u + v) <= 12.
 * @param t X-coordinate order of the the Hermite Coulomb integral.
 * @param u Y-coordinate order of the the Hermite Coulomb integral.
 * @param v Z-coordinate order of the the Hermite Coulomb integral.
 * @param p Reduced exponent of the pair of Hermite Gaussians.
 * @param X X-component of the vector from the first Hermite Gaussian to the second.
 * @param Y Y-component of the vector from the first Hermite Gaussian to the second.
 * @param Z Z-component of the vector from the first Hermite Gaussian to the second.
 * @return double Value of the Hermite Coulomb integral.
 */
double IntegralsCoulomb2C::HermiteCoulomb(const int t, const int u, const int v, const double p, const double X, const double Y, const double Z){

    double arg = p*(X*X + Y*Y + Z*Z);
    switch(v)
    {
    case 0:  {
        switch(u)
        {
        case 0:  {
            switch(t)
            {
            case 0:  {
                return Boysfun(0, arg);
            } 
            case 1:  {
                return X*Rn000(1,p,arg);
            }
            case 2:  {
                return Rn000(1,p,arg) + X*X*Rn000(2,p,arg);
            }
            case 3:  {
                return X*(3*Rn000(2,p,arg) + X*X*Rn000(3,p,arg));
            }
            case 4:  {
                return 3*Rn000(2,p,arg) + X*X*(6*Rn000(3,p,arg) + X*X*Rn000(4,p,arg));
            }
            case 5:  {
                double X2 = X*X;
                return X*(15*Rn000(3,p,arg) + X2*(10*Rn000(4,p,arg) + X2*Rn000(5,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                return 15*Rn000(3,p,arg) + X2*(45*Rn000(4,p,arg) + X2*(15*Rn000(5,p,arg) + X2*Rn000(6,p,arg)));
            } 
            case 7:  {
                double X2 = X*X;
                return X*(105*Rn000(4,p,arg) + X2*(105*Rn000(5,p,arg) + X2*(21*Rn000(6,p,arg) + X2*Rn000(7,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                return 105*Rn000(4,p,arg) + X2*(420*Rn000(5,p,arg) + X2*(210*Rn000(6,p,arg) + X2*(28*Rn000(7,p,arg) + X2*Rn000(8,p,arg))));
            }
            case 9:  {
                double X2 = X*X;
                return X*(945*Rn000(5,p,arg) + X2*(1260*Rn000(6,p,arg) + X2*(378*Rn000(7,p,arg) + X2*(36*Rn000(8,p,arg) + X2*Rn000(9,p,arg)))));
            }
            case 10: {
                double X2 = X*X;
                return 945*Rn000(5,p,arg) + X2*(4725*Rn000(6,p,arg) + X2*(3150*Rn000(7,p,arg) + X2*(630*Rn000(8,p,arg) + X2*(45*Rn000(9,p,arg) + X2*Rn000(10,p,arg)))));
            }
            case 11: {
                double X2 = X*X;
                return X*(10395*Rn000(6,p,arg) + X2*(17325*Rn000(7,p,arg) + X2*(6930*Rn000(8,p,arg) + X2*(990*Rn000(9,p,arg) + X2*(55*Rn000(10,p,arg) + X2*Rn000(11,p,arg))))));
            }
            case 12: {
                double X2 = X*X;
                return 10395*Rn000(6,p,arg) + X2*(62370*Rn000(7,p,arg) + X2*(51975*Rn000(8,p,arg) + X2*(13860*Rn000(9,p,arg) + X2*(1485*Rn000(10,p,arg) + X2*(66*Rn000(11,p,arg) + X2*Rn000(12,p,arg))))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=0,v=0)");
            }
            }
        }
        case 1:  {
            switch(t)
            {
            case 1:  {
                return X*Y*Rn000(2,p,arg);
            }
            case 2:  {
                return Y*(Rn000(2,p,arg) + X*X*Rn000(3,p,arg));
            }
            case 3:  {
                return X*Y*(3*Rn000(3,p,arg) + X*X*Rn000(4,p,arg));
            }
            case 4:  {
                return Y*(3*Rn000(3,p,arg) + X*X*(6*Rn000(4,p,arg) + X*X*Rn000(5,p,arg)));
            }
            case 5:  {
                return X*Y*(15*Rn000(4,p,arg) + X*X*(10*Rn000(5,p,arg) + X*X*Rn000(6,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                return Y*(15*Rn000(4,p,arg) + X2*(45*Rn000(5,p,arg) + X2*(15*Rn000(6,p,arg) + X2*Rn000(7,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                return X*Y*(105*Rn000(5,p,arg) + X2*(105*Rn000(6,p,arg) + X2*(21*Rn000(7,p,arg) + X2*Rn000(8,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                return Y*(105*Rn000(5,p,arg) + X2*(420*Rn000(6,p,arg) + X2*(210*Rn000(7,p,arg) + X2*(28*Rn000(8,p,arg) + X2*Rn000(9,p,arg)))));
            }
            case 9:  {
                double X2 = X*X;
                return X*Y*(945*Rn000(6,p,arg) + X2*(1260*Rn000(7,p,arg) + X2*(378*Rn000(8,p,arg) + X2*(36*Rn000(9,p,arg) + X2*Rn000(10,p,arg)))));
            }
            case 10: {
                double X2 = X*X;
                return Y*(945*Rn000(6,p,arg) + X2*(4725*Rn000(7,p,arg) + X2*(3150*Rn000(8,p,arg) + X2*(630*Rn000(9,p,arg) + X2*(45*Rn000(10,p,arg) + X2*Rn000(11,p,arg))))));
            }
            case 11: {
                double X2 = X*X;
                return X*Y*(10395*Rn000(7,p,arg) + X2*(17325*Rn000(8,p,arg) + X2*(6930*Rn000(9,p,arg) + X2*(990*Rn000(10,p,arg) + X2*(55*Rn000(11,p,arg) + X2*Rn000(12,p,arg))))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=1,v=0)");
            }
            }
        }
        case 2:  {
            switch(t)
            {
            case 2:  {
                return Rn000(2,p,arg) + (X*X + Y*Y)*Rn000(3,p,arg) + X*X*Y*Y*Rn000(4,p,arg);
            }
            case 3:  {
                return X*(3*Rn000(3,p,arg) + (X*X + 3*Y*Y)*Rn000(4,p,arg) + X*X*Y*Y*Rn000(5,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return 3*Rn000(3,p,arg) + 3*(2*X2 + Y2)*Rn000(4,p,arg) + X2*((X2 + 6*Y2)*Rn000(5,p,arg) + X2*Y2*Rn000(6,p,arg));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*(15*Rn000(4,p,arg) + (10*X2 + 15*Y2)*Rn000(5,p,arg) + X2*((X2 + 10*Y2)*Rn000(6,p,arg) + X2*Y2*Rn000(7,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return 15*Rn000(4,p,arg) + 15*(3*X2 + Y2)*Rn000(5,p,arg) + X2*(15*(X2 + 3*Y2)*Rn000(6,p,arg) + X2*((X2 + 15*Y2)*Rn000(7,p,arg) + X2*Y2*Rn000(8,p,arg)));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*(105*Rn000(5,p,arg) + 105*(X2 + Y2)*Rn000(6,p,arg) + X2*( 21*(X2 + 5*Y2)*Rn000(7,p,arg) + X2*((X2 + 21*Y2)*Rn000(8,p,arg) + X2*Y2*Rn000(9,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return 105*Rn000(5,p,arg) + (420*X2 + 105*Y2)*Rn000(6,p,arg) + X2*((210*X2 + 420*Y2)*Rn000(7,p,arg) + X2*((28*X2 + 210*Y2)*Rn000(8,p,arg) + X2*((X2 + 28*Y2)*Rn000(9,p,arg) + X2*Y2*Rn000(10,p,arg))));
            }
            case 9:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*(945*Rn000(6,p,arg) + (1260*X2 + 945*Y2)*Rn000(7,p,arg) + X2*((378*X2 + 1260*Y2)*Rn000(8,p,arg) + X2*((36*X2 + 378*Y2)*Rn000(9,p,arg) + X2*((X2 + 36*Y2)*Rn000(10,p,arg) + X2*Y2*Rn000(11,p,arg)))));
            }
            case 10: {
                double X2 = X*X;
                double Y2 = Y*Y;
                return 945*Rn000(6,p,arg) + (4725*X2 + 945*Y2)*Rn000(7,p,arg) + X2*((3150*X2 + 4725*Y2)*Rn000(8,p,arg) + X2*((630*X2 + 3150*Y2)*Rn000(9,p,arg) + X2*((45*X2 + 630*Y2)*Rn000(10,p,arg) + X2*((X2 + 45*Y2)*Rn000(11,p,arg) + X2*Y2*Rn000(12,p,arg)))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=2,v=0)");
            }
            }
        }
        case 3:  {
            switch(t)
            {
            case 3:  {
                return X*Y*(9*Rn000(4,p,arg) + 3*(X*X + Y*Y)*Rn000(5,p,arg) + X*X*Y*Y*Rn000(6,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*(9*Rn000(4,p,arg) + 3*(6*X2 + Y2)*Rn000(5,p,arg) + X2*((3*X2 + 6*Y2)*Rn000(6,p,arg) + X2*Y2*Rn000(7,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Y*(45*Rn000(5,p,arg) + 15*(2*X2 + Y2)*Rn000(6,p,arg) + X2*((3*X2 + 10*Y2)*Rn000(7,p,arg) + X2*Y2*Rn000(8,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*(45*Rn000(5,p,arg) + (135*X2 + 15*Y2)*Rn000(6,p,arg) + X2*(45*(X2 + Y2)*Rn000(7,p,arg) + X2*((3*X2 + 15*Y2)*Rn000(8,p,arg) + X2*Y2*Rn000(9,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Y*(315*Rn000(6,p,arg) + (315*X2 + 105*Y2)*Rn000(7,p,arg) + X2*((63*X2 + 105*Y2)*Rn000(8,p,arg) + X2*((3*X2 + 21*Y2)*Rn000(9,p,arg) + X2*Y2*Rn000(10,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*(315*Rn000(6,p,arg) + (1260*X2 + 105*Y2)*Rn000(7,p,arg) + X2*((630*X2 + 420*Y2)*Rn000(8,p,arg) + X2*((84*X2 + 210*Y2)*Rn000(9,p,arg) + X2*((3*X2 + 28*Y2)*Rn000(10,p,arg) + X2*Y2*Rn000(11,p,arg)))));
            }
            case 9:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Y*(2835*Rn000(7,p,arg) + (3780*X2 + 945*Y2)*Rn000(8,p,arg) + X2*((1134*X2 + 1260*Y2)*Rn000(9,p,arg) + X2*((108*X2 + 378*Y2)*Rn000(10,p,arg) + X2*((3*X2 + 36*Y2)*Rn000(11,p,arg) + X2*Y2*Rn000(12,p,arg)))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=3,v=0)");
            }
            }
        }
        case 4:  {
            switch(t)
            {
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return 9*Rn000(4,p,arg) + 18*(X2 + Y2)*Rn000(5,p,arg) + 3*(X2*X2 + Y2*Y2 + 12*X2Y2)*Rn000(6,p,arg) + X2Y2*(6*(X2 + Y2)*Rn000(7,p,arg) + X2Y2*Rn000(8,p,arg));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*(45*Rn000(5,p,arg) + (30*X2 + 90*Y2)*Rn000(6,p,arg) + (3*X2*X2 + 60*X2Y2 + 15*Y2*Y2)*Rn000(7,p,arg) + X2Y2*((6*X2 + 10*Y2)*Rn000(8,p,arg) + X2Y2*Rn000(9,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return 45*Rn000(5,p,arg) + (135*X2 + 90*Y2)*Rn000(6,p,arg) + (45*X2*X2 + 270*X2Y2 + 15*Y2*Y2)*Rn000(7,p,arg) + X2*((3*X2*X2 + 90*X2Y2 + 45*Y2*Y2)*Rn000(8,p,arg) + X2Y2*((6*X2 + 15*Y2)*Rn000(9,p,arg) + X2Y2*Rn000(10,p,arg)));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*(315*Rn000(6,p,arg) + (315*X2 + 630*Y2)*Rn000(7,p,arg) + (63*X2*X2 + 630*X2Y2 + 105*Y2*Y2)*Rn000(8,p,arg) + X2*((3*X2*X2 + 126*X2Y2 + 105*Y2*Y2)*Rn000(9,p,arg) + X2Y2*((6*X2 + 21*Y2)*Rn000(10,p,arg) + X2Y2*Rn000(11,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double X2Y2 = X2*Y2;
                return 315*Rn000(6,p,arg) + (1260*X2 + 630*Y2)*Rn000(7,p,arg) + (630*X4 + 2520*X2Y2 + 105*Y4)*Rn000(8,p,arg) + X2*((84*X4 + 1260*X2Y2 + 420*Y4)*Rn000(9,p,arg) + X2*((3*X4 + 168*X2Y2 + 210*Y4)*Rn000(10,p,arg) + X2Y2*((6*X2 + 28*Y2)*Rn000(11,p,arg) + X2Y2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=4,v=0)");
            }
            }
        }
        case 5:  {
            switch(t)
            {
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*Y*(225*Rn000(6,p,arg) + 150*(X2 + Y2)*Rn000(7,p,arg) + (15*X2*X2 + 100*X2Y2 + 15*Y2*Y2)*Rn000(8,p,arg) + X2Y2*(10*(X2 + Y2)*Rn000(9,p,arg) + X2Y2*Rn000(10,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return Y*(225*Rn000(6,p,arg) + (675*X2 + 150*Y2)*Rn000(7,p,arg) + (225*X2*X2 + 450*X2Y2 + 15*Y2*Y2)*Rn000(8,p,arg) + X2*((15*X2*X2 + 150*X2Y2 + 45*Y2*Y2)*Rn000(9,p,arg) + X2Y2*((10*X2 + 15*Y2)*Rn000(10,p,arg) + X2Y2*Rn000(11,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*Y*(1575*Rn000(7,p,arg) + (1575*X2 + 1050*Y2)*Rn000(8,p,arg) + (315*X2*X2 + 1050*X2Y2 + 105*Y2*Y2)*Rn000(9,p,arg) + X2*((15*X2*X2 + 210*X2Y2 + 105*Y2*Y2)*Rn000(10,p,arg) + X2Y2*((10*X2 + 21*Y2)*Rn000(11,p,arg) + X2Y2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=5,v=0)");
            }
            }
        }
        case 6:  {
            switch(t)
            {
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double X2Y2 = X2*Y2;
                return 225*Rn000(6,p,arg) + 675*(X2 + Y2)*Rn000(7,p,arg) + (225*X4 + 2025*X2Y2 + 225*Y4)*Rn000(8,p,arg) + (15*X2*X4 + 675*X2*Y2*(X2 + Y2) + 15*Y4*Y2)*Rn000(9,p,arg) + X2Y2*((45*X4 + 225*X2Y2 + 45*Y4)*Rn000(10,p,arg) + X2Y2*(15*(X2 + Y2)*Rn000(11,p,arg) + X2Y2*Rn000(12,p,arg)));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=6,v=0)");
            }
            }
        }
        default:  {
            throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with v=0)");
        }
        }
    } 
    case 1:  {
        switch(u)
        {
        case 1:  {
            switch(t)
            {
            case 1:  {
                return X*Y*Z*Rn000(3,p,arg);
            }
            case 2:  {
                return Y*Z*(Rn000(3,p,arg) + X*X*Rn000(4,p,arg));
            }
            case 3:  {
                return X*Y*Z*(3*Rn000(4,p,arg) + X*X*Rn000(5,p,arg));
            }
            case 4:  {
                return Y*Z*(3*Rn000(4,p,arg) + X*X*(6*Rn000(5,p,arg) + X*X*Rn000(6,p,arg)));
            }
            case 5:  {
                return X*Y*Z*(15*Rn000(5,p,arg) + X*X*(10*Rn000(6,p,arg) + X*X*Rn000(7,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                return Y*Z*(15*Rn000(5,p,arg) + X2*(45*Rn000(6,p,arg) + X2*(15*Rn000(7,p,arg) + X2*Rn000(8,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                return X*Y*Z*(105*Rn000(6,p,arg) + X2*(105*Rn000(7,p,arg) + X2*(21*Rn000(8,p,arg) + X2*Rn000(9,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                return Y*Z*(105*Rn000(6,p,arg) + X2*(420*Rn000(7,p,arg) + X2*(210*Rn000(8,p,arg) + X2*(28*Rn000(9,p,arg) + X2*Rn000(10,p,arg)))));
            }
            case 9:  {
                double X2 = X*X;
                return X*Y*Z*(945*Rn000(7,p,arg) + X2*(1260*Rn000(8,p,arg) + X2*(378*Rn000(9,p,arg) + X2*(36*Rn000(10,p,arg) + X2*Rn000(11,p,arg)))));
            }
            case 10:  {
                double X2 = X*X;
                return Y*Z*(945*Rn000(7,p,arg) + X2*(4725*Rn000(8,p,arg) + X2*(3150*Rn000(9,p,arg) + X2*(630*Rn000(10,p,arg) + X2*(45*Rn000(11,p,arg) + X2*Rn000(12,p,arg))))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=1,v=1)");
            }
            }
        }
        case 2:  {
            switch(t)
            {
            case 2:  {
                return Z*(Rn000(3,p,arg) + (X*X + Y*Y)*Rn000(4,p,arg) + X*X*Y*Y*Rn000(5,p,arg));
            }
            case 3:  {
                return X*Z*(3*Rn000(4,p,arg) + (X*X + 3*Y*Y)*Rn000(5,p,arg) + X*X*Y*Y*Rn000(6,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Z*(3*Rn000(4,p,arg) + 3*(2*X2 + Y2)*Rn000(5,p,arg) + X2*((X2 + 6*Y2)*Rn000(6,p,arg) + X2*Y2*Rn000(7,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Z*(15*Rn000(5,p,arg) + (10*X2 + 15*Y2)*Rn000(6,p,arg) + X2*((X2 + 10*Y2)*Rn000(7,p,arg) + X2*Y2*Rn000(8,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Z*(15*Rn000(5,p,arg) + (45*X2 + 15*Y2)*Rn000(6,p,arg) + X2*((15*X2 + 45*Y2)*Rn000(7,p,arg) + X2*((X2 + 15*Y2)*Rn000(8,p,arg) + X2*Y2*Rn000(9,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Z*(105*Rn000(6,p,arg) + 105*(X2 + Y2)*Rn000(7,p,arg) + X2*((21*X2 + 105*Y2)*Rn000(8,p,arg) + X2*((X2 + 21*Y2)*Rn000(9,p,arg) + X2*Y2*Rn000(10,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Z*(105*Rn000(6,p,arg) + (420*X2 + 105*Y2)*Rn000(7,p,arg) + X2*((210*X2 + 420*Y2)*Rn000(8,p,arg) + X2*((28*X2 + 210*Y2)*Rn000(9,p,arg) + X2*((X2 + 28*Y2)*Rn000(10,p,arg) + X2*Y2*Rn000(11,p,arg)))));
            }
            case 9:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Z*(945*Rn000(7,p,arg) + (1260*X2 + 945*Y2)*Rn000(8,p,arg) + X2*((378*X2 + 1260*Y2)*Rn000(9,p,arg) + X2*((36*X2 + 378*Y2)*Rn000(10,p,arg) + X2*((X2 + 36*Y2)*Rn000(11,p,arg) + X2*Y2*Rn000(12,p,arg)))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=2,v=1)");
            }
            }
        }
        case 3:  {
            switch(t)
            {
            case 3:  {
                double XY = X*Y;
                return XY*Z*(9*Rn000(5,p,arg) + 3*(X*X + Y*Y)*Rn000(6,p,arg) + XY*XY*Rn000(7,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*Z*(9*Rn000(5,p,arg) + 3*(6*X2 + Y2)*Rn000(6,p,arg) + X2*(3*(X2 + 2*Y2)*Rn000(7,p,arg) + X2*Y2*Rn000(8,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Y*Z*(45*Rn000(6,p,arg) + (30*X2 + 15*Y2)*Rn000(7,p,arg) + X2*((3*X2 + 10*Y2)*Rn000(8,p,arg) + X2*Y2*Rn000(9,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*Z*(45*Rn000(6,p,arg) + (135*X2 + 15*Y2)*Rn000(7,p,arg) + X2*(45*(X2 + Y2)*Rn000(8,p,arg) + X2*((3*X2 + 15*Y2)*Rn000(9,p,arg) + X2*Y2*Rn000(10,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return X*Y*Z*(315*Rn000(7,p,arg) + (315*X2 + 105*Y2)*Rn000(8,p,arg) + X2*((63*X2 + 105*Y2)*Rn000(9,p,arg) + X2*((3*X2 + 21*Y2)*Rn000(10,p,arg) + X2*Y2*Rn000(11,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                return Y*Z*(315*Rn000(7,p,arg) + (1260*X2 + 105*Y2)*Rn000(8,p,arg) + X2*((630*X2 + 420*Y2)*Rn000(9,p,arg) + X2*((84*X2 + 210*Y2)*Rn000(10,p,arg) + X2*((3*X2 + 28*Y2)*Rn000(11,p,arg) + X2*Y2*Rn000(12,p,arg)))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=3,v=1)");
            }
            }
        }
        case 4:  {
            switch(t)
            {
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return Z*(9*Rn000(5,p,arg) + 18*(X2 + Y2)*Rn000(6,p,arg) + 3*(X2*X2 + 12*X2Y2 + Y2*Y2)*Rn000(7,p,arg) + X2Y2*(6*(X2 + Y2)*Rn000(8,p,arg) + X2Y2*Rn000(9,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*Z*(45*Rn000(6,p,arg) + (30*X2 + 90*Y2)*Rn000(7,p,arg) + (3*X2*X2 + 60*X2Y2 + 15*Y2*Y2)*Rn000(8,p,arg) + X2Y2*((6*X2 + 10*Y2)*Rn000(9,p,arg) + X2Y2*Rn000(10,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return Z*(45*Rn000(6,p,arg) + (135*X2 + 90*Y2)*Rn000(7,p,arg) + (45*X2*X2 + 270*X2Y2 + 15*Y2*Y2)*Rn000(8,p,arg) + X2*((3*X2*X2 + 90*X2Y2 + 45*Y2*Y2)*Rn000(9,p,arg) + X2Y2*((6*X2 + 15*Y2)*Rn000(10,p,arg) + X2Y2*Rn000(11,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*Z*(315*Rn000(7,p,arg) + (315*X2 + 630*Y2)*Rn000(8,p,arg) + (63*X2*X2 + 630*X2Y2 + 105*Y2*Y2)*Rn000(9,p,arg) + X2*((3*X2*X2 + 126*X2Y2 + 105*Y2*Y2)*Rn000(10,p,arg) + X2Y2*((6*X2 + 21*Y2)*Rn000(11,p,arg) + X2Y2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=4,v=1)");
            }
            }
        }
        case 5:  {
            switch(t)
            {
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return X*Y*Z*(225*Rn000(7,p,arg) + 150*(X2 + Y2)*Rn000(8,p,arg) + (15*X2*X2 + 100*X2Y2 + 15*Y2*Y2)*Rn000(9,p,arg) + X2Y2*(10*(X2 + Y2)*Rn000(10,p,arg) + X2Y2*Rn000(11,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double X2Y2 = X2*Y2;
                return Y*Z*(225*Rn000(7,p,arg) + (675*X2 + 150*Y2)*Rn000(8,p,arg) + (225*X2*X2 + 450*X2Y2 + 15*Y2*Y2)*Rn000(9,p,arg) + X2*((15*X2*X2 + 150*X2Y2 + 45*Y2*Y2)*Rn000(10,p,arg) + X2Y2*((10*X2 + 15*Y2)*Rn000(11,p,arg) + X2Y2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=5,v=1)");
            }
            }
        }
        default:  {
            throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with v=1)");
        }
        }
    } 
    case 2:  {
        switch(u)
        {
        case 2:  {
            switch(t)
            {
            case 2:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                return Rn000(3,p,arg) + (X2 + Y2 + Z2)*Rn000(4,p,arg) + (X2*(Y2 + Z2) + Y2*Z2)*Rn000(5,p,arg) + X2*Y2*Z2*Rn000(6,p,arg);
            }
            case 3:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                return X*(3*Rn000(4,p,arg) + (X2 + 3*(Y2 + Z2))*Rn000(5,p,arg) + (X2*(Y2 + Z2) + 3*Y2*Z2)*Rn000(6,p,arg) + X2*Y2*Z2*Rn000(7,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return 3*Rn000(4,p,arg) + 3*(2*X2 + Y2 + Z2)*Rn000(5,p,arg) + (X2*(X2 + 6*(Y2 + Z2)) + 3*Y2Z2)*Rn000(6,p,arg) + X2*((6*Y2Z2 + X2*(Y2 + Z2))*Rn000(7,p,arg) + X2*Y2Z2*Rn000(8,p,arg));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return X*(15*Rn000(5,p,arg) + (10*X2 + 15*(Y2 + Z2))*Rn000(6,p,arg) + (10*X2*(Y2 + Z2) + X2*X2 + 15*Y2Z2)*Rn000(7,p,arg) + X2*((X2*(Y2 + Z2) + 10*Y2Z2)*Rn000(8,p,arg) + X2*Y2Z2*Rn000(9,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                double X2YpZ = X2*(Y2 + Z2);
                return 15*Rn000(5,p,arg) + 15*(3*X2 + Y2 + Z2)*Rn000(6,p,arg) + 15*(Y2Z2 + 3*X2YpZ + X2*X2)*Rn000(7,p,arg) + X2*((15*X2YpZ + X2*X2 + 45*Y2Z2)*Rn000(8,p,arg) + X2*((X2YpZ + 15*Y2Z2)*Rn000(9,p,arg) + X2*Y2Z2*Rn000(10,p,arg)));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                double X2YpZ = X2*(Y2 + Z2);
                return X*(105*Rn000(6,p,arg) + 105*(X2 + Y2 + Z2)*Rn000(7,p,arg) + 21*(5*(X2YpZ + Y2Z2) + X2*X2)*Rn000(8,p,arg) + X2*((21*X2YpZ + X2*X2 + 105*Y2Z2)*Rn000(9,p,arg) + X2*((X2YpZ + 21*Y2Z2)*Rn000(10,p,arg) + X2*Y2Z2*Rn000(11,p,arg))));
            }
            case 8:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y2Z2 = Y2*Z2;
                double X2YpZ = X2*(Y2 + Z2);
                return 105*Rn000(6,p,arg) + 105*(4*X2 + Y2 + Z2)*Rn000(7,p,arg) + (105*Y2Z2 + 420*X2YpZ + 210*X4)*Rn000(8,p,arg) + X2*((210*X2YpZ + 28*X4 + 420*Y2Z2)*Rn000(9,p,arg) + X2*((28*X2YpZ + X4 + 210*Y2Z2)*Rn000(10,p,arg) + X2*((X2YpZ + 28*Y2Z2)*Rn000(11,p,arg) + X2*Y2Z2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=2,v=2)");
            }
            }
        }
        case 3:  {
            switch(t)
            {
            case 3:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                return X*Y*(9*Rn000(5,p,arg) + 3*(X2 + Y2 + 3*Z2)*Rn000(6,p,arg) + (X2*Y2 + 3*Z2*(X2 + Y2))*Rn000(7,p,arg) + X2*Y2*Z2*Rn000(8,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return Y*(9*Rn000(5,p,arg) + (18*X2 + 3*Y2 + 9*Z2)*Rn000(6,p,arg) + (3*Y2Z2 + 6*X2*(Y2 + 3*Z2) + 3*X2*X2)*Rn000(7,p,arg) + X2*((X2*(Y2 + 3*Z2) + 6*Y2Z2)*Rn000(8,p,arg) + X2*Y2Z2*Rn000(9,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                double Y2pZ = (Y2 + 3*Z2);
                return X*Y*(45*Rn000(6,p,arg) + (30*X2 + 15*Y2pZ)*Rn000(7,p,arg) + (10*X2*Y2pZ + 3*X2*X2 + 15*Y2Z2)*Rn000(8,p,arg) + X2*((X2*Y2pZ + 10*Y2Z2)*Rn000(9,p,arg) + X2*Y2Z2*Rn000(10,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                double X2Y2pZ = X2*(Y2 + 3*Z2);
                return Y*(45*Rn000(6,p,arg) + (135*X2 + 15*Y2 + 45*Z2)*Rn000(7,p,arg) + 15*(Y2Z2 + 3*(X2Y2pZ + X2*X2))*Rn000(8,p,arg) + X2*((15*X2Y2pZ + 3*X2*X2 + 45*Y2Z2)*Rn000(9,p,arg) + X2*((X2Y2pZ + 15*Y2Z2)*Rn000(10,p,arg) + X2*Y2Z2*Rn000(11,p,arg))));
            }
            case 7:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                double Y2pZ = (Y2 + 3*Z2);
                return X*Y*(315*Rn000(7,p,arg) + (315*X2 + 105*Y2pZ)*Rn000(8,p,arg) + (105*X2*Y2pZ + 63*X2*X2 + 105*Y2Z2)*Rn000(9,p,arg) + X2*((21*X2*Y2pZ + 3*X2*X2 + 105*Y2Z2)*Rn000(10,p,arg) + X2*((X2*Y2pZ + 21*Y2Z2)*Rn000(11,p,arg) + X2*Y2Z2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=3,v=2)");
            }
            }
        }
        case 4:  {
            switch(t)
            {
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double Y2Z2 = Y2*Z2;
                return 9*Rn000(5,p,arg) + 9*(2*(X2 + Y2) + Z2)*Rn000(6,p,arg) + (18*X2*(2*Y2 + Z2) + 18*Y2Z2 + 3*(X4 + Y4))*Rn000(7,p,arg) + (3*X4*(2*Y2 + Z2) + 3*Y4*Z2 + 6*X2*(Y4 + 6*Y2Z2))*Rn000(8,p,arg) + X2*Y2*((X2*(Y2 + 6*Z2) + 6*Y2Z2)*Rn000(9,p,arg) + X2*Y2Z2*Rn000(10,p,arg));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double Y2Z2 = Y2*Z2;
                double Y22Z2 = 2*Y2 + Z2;
                return X*(45*Rn000(6,p,arg) + (45*Y22Z2 + 30*X2)*Rn000(7,p,arg) + (30*X2*Y22Z2 + 15*Y2*(Y2 + 6*Z2) + 3*X4)*Rn000(8,p,arg) + (3*X4*Y22Z2 + 10*X2*(Y4 + 6*Y2Z2) + 15*Y4*Z2)*Rn000(9,p,arg) + X2*Y2*((X2*(Y2 + 6*Z2) + 10*Y2Z2)*Rn000(10,p,arg) + X2*Y2Z2*Rn000(11,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double X2Y2 = X2*Y2;
                double Y2Z2 = Y2*Z2;
                double Y22Z2 = 2*Y2 + Z2;
                double Y2p6Z2 = Y2 + 6*Z2;
                return 45*Rn000(6,p,arg) + 45*(3*X2 + Y22Z2)*Rn000(7,p,arg) + 15*(9*X2*Y22Z2 + 6*Y2Z2 + 3*X4 + Y4)*Rn000(8,p,arg) + 3*(15*X4*Y22Z2 + 5*Y4*Z2 + 15*X2Y2*Y2p6Z2 + X4*X2)*Rn000(9,p,arg) + X2*(3*(X4*Y22Z2 + 5*X2Y2*Y2p6Z2 + 15*Y4*Z2)*Rn000(10,p,arg) + X2Y2*((X2*Y2p6Z2 + 15*Y2Z2)*Rn000(11,p,arg) + X2*Y2Z2*Rn000(12,p,arg)));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=4,v=2)");
            }
            }
        }
        case 5:  {
            switch(t)
            {
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double Y22p3Z2 = 2*Y2 + 3*Z2;
                double Y2p10Z2 = Y2 + 10*Z2;
                return X*Y*(225*Rn000(7,p,arg) + 75*(2*X2 + Y22p3Z2)*Rn000(8,p,arg) + (15*(X4 + Y2*Y2p10Z2) + 50*X2*Y22p3Z2)*Rn000(9,p,arg) + 5*(X4*Y22p3Z2 + 2*X2*Y2*Y2p10Z2 + 3*Y4*Z2)*Rn000(10,p,arg) + X2*Y2*((X2*Y2p10Z2 + 10*Y2*Z2)*Rn000(11,p,arg) + X2*Y2*Z2*Rn000(12,p,arg)));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=5,v=2)");
            }
            }
        }
        default:  {
            throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with v=2)");
        }
        }
    }
    case 3:  {
        switch(u)
        {
        case 3:  {
            switch(t)
            {
            case 3:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                return X*Y*Z*(27*Rn000(6,p,arg) + 9*(X2 + Y2 + Z2)*Rn000(7,p,arg) + 3*(X2*Y2 + X2*Z2 + Y2*Z2)*Rn000(8,p,arg) + X2*Y2*Z2*Rn000(9,p,arg));
            }
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return Y*Z*(27*Rn000(6,p,arg) + 9*(6*X2 + Y2 + Z2)*Rn000(7,p,arg) + 3*(Y2Z2 + 3*X2*(X2 + 2*(Y2 + Z2)))*Rn000(8,p,arg) + X2*(3*(X2*(Y2 + Z2) + 2*Y2Z2)*Rn000(9,p,arg) + X2*Y2Z2*Rn000(10,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return X*Y*Z*(135*Rn000(7,p,arg) + 45*(2*X2 + Y2 + Z2)*Rn000(8,p,arg) + (30*X2*(Y2 + Z2) + 9*X2*X2 + 15*Y2Z2)*Rn000(9,p,arg) + X2*((3*X2*(Y2 + Z2) + 10*Y2Z2)*Rn000(10,p,arg) + X2*Y2Z2*Rn000(11,p,arg)));
            }
            case 6:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double Y2Z2 = Y2*Z2;
                return Y*Z*(135*Rn000(7,p,arg) + 45*(9*X2 + Y2 + Z2)*Rn000(8,p,arg) + 15*(Y2Z2 + 9*X2*(X2 + Y2 + Z2))*Rn000(9,p,arg) + X2*(9*(5*(X2*(Y2 + Z2) + Y2Z2) + X2*X2)*Rn000(10,p,arg) + X2*(3*(X2*(Y2 + Z2) + 5*Y2Z2)*Rn000(11,p,arg) + X2*Y2Z2*Rn000(12,p,arg))));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=3,v=3)");
            }
            }
        }
        case 4:  {
            switch(t)
            {
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X2Y2 = X2*Y2;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                return Z*(27*Rn000(6,p,arg) + 9*(6*(X2 + Y2) + Z2)*Rn000(7,p,arg) + 9*(2*(X2 + Y2)*Z2 + 12*X2Y2 + X4 + Y4)*Rn000(8,p,arg) + 3*((X4 + Y4)*Z2 + 6*(X2*Y4 + 2*X2Y2*Z2 + X4*Y2))*Rn000(9,p,arg) + X2Y2*(3*(X2Y2 + 2*(X2 + Y2)*Z2)*Rn000(10,p,arg) + X2Y2*Z2*Rn000(11,p,arg)));
            }
            case 5:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X2Y2 = X2*Y2;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double Y26pZ2 = 6*Y2 + Z2;
                return X*Z*(135*Rn000(7,p,arg) + 45*(2*X2 + Y26pZ2)*Rn000(8,p,arg) + (30*X2*Y26pZ2 + 9*X4 + 45*(Y4 + 2*Y2*Z2))*Rn000(9,p,arg) + 3*(10*X2Y2*(Y2 + 2*Z2) + X4*Y26pZ2 + 5*Y4*Z2)*Rn000(10,p,arg) + X2Y2*((3*X2*(Y2 + 2*Z2) + 10*Y2*Z2)*Rn000(11,p,arg) + X2Y2*Z2*Rn000(12,p,arg)));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=4,v=3)");
            }
            }
        }
        default:  {
            throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with v=3)");
        }
        }
    }
    case 4:  {
        switch(u)
        {
        case 4:  {
            switch(t)
            {
            case 4:  {
                double X2 = X*X;
                double Y2 = Y*Y;
                double Z2 = Z*Z;
                double X4 = X2*X2;
                double Y4 = Y2*Y2;
                double Z4 = Z2*Z2;
                double X2Y2Z2 = X2*Y2*Z2;
                double X2Y2Z2combs = X2*Y2 + X2*Z2 + Y2*Z2;
                return 27*Rn000(6,p,arg) + 54*(X2 + Y2 + Z2)*Rn000(7,p,arg) + 9*(12*X2Y2Z2combs + X4 + Y4 + Z4)*Rn000(8,p,arg) + (18*(X4*(Y2 + Z2) + Y4*(X2 + Z2) + Z4*(X2 + Y2)) + 216*X2Y2Z2)*Rn000(9,p,arg) + 3*((X4*Y4 + X4*Z4 + Y4*Z4) + 12*X2Y2Z2*(X2 + Y2 + Z2))*Rn000(10,p,arg) + X2Y2Z2*(6*X2Y2Z2combs*Rn000(11,p,arg) + X2Y2Z2*Rn000(12,p,arg));
            }
            default: {
                throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with u=4,v=4)");
            }
            }
        }
        default:  {
            throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, with v=4)");
        }
        }
    }
    default:  {
        throw std::invalid_argument("IntegralsCoulomb2C::HermiteCoulomb error: the R^{0}_{t,u,v} coefficients are being evaluated for t+u+v > 12 (in particular, v>=5)");
    }
    }

}

/**
 * Compute the (normalized) vector with the charges of each orbital (fixed l,m) in the AUX basis, and save it to charges_intName.chg. 
 * Only the s-type GTF have finite charge. This is currently not used.
 * @return void.
 */
void IntegralsCoulomb2C::computeCharges(const std::string& intName){

    arma::colvec charges = arma::zeros<arma::colvec>(dimMat_AUX);
    int L, nG;
    double exponent;
    for(uint32_t orb = 0; orb < dimMat_AUX; orb++){
        L = orbitals_info_int_AUX_[orb][2];
        if(L == 0){
            nG = orbitals_info_int_AUX_[orb][4];
            for(int gaussC = 0; gaussC < nG; gaussC++){
                exponent = orbitals_info_real_AUX[orb][2*gaussC + 3];
                charges(orb) = FAC3_AUX[orb][gaussC]*std::pow(PI/exponent,1.5);
            }
            charges(orb) *= FAC12_AUX[orb];
        }
    }
    charges = normalise(charges,2);
    charges.save(IntFiles_Dir + "charges_" + intName + ".chg", arma::arma_ascii);

}

}
