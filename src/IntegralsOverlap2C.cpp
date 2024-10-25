#include "xatu/IntegralsOverlap2C.hpp"

namespace xatu {

/**
 * Constructor that copies a pre-initialized IntegralsBase object.
 * @param IntBase IntegralsBase object.
 * @param tol Threshold tolerance for the integrals: only entries > 10^-tol are stored.
 * @param nR Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed.
 * @param intName Name of the file where the 2-center overlap matrices will be stored as a vector (o2Mat_intName.o2c).
 * @param basis_id True => SCF basis, False => Auxiliary basis.
 */
IntegralsOverlap2C::IntegralsOverlap2C(const IntegralsBase& IntBase, const int tol, const uint32_t nR, const std::string& intName, const bool basis_id) : IntegralsBase{IntBase} {

    overlap2Cfun(tol, nR, intName, basis_id);

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
void IntegralsOverlap2C::overlap2Cfun(const int tol, const uint32_t nR, const std::string& intName, const bool basis_id){

#pragma omp declare reduction (merge : std::vector<std::array<double,4>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

const double PIpow = std::pow(PI,1.5);
arma::mat combs;
arma::mat RlistAU = ANG2AU*generateRlist(nR, combs, "Overlap2C");  //convert Bravais vectors from Angstrom to atomic units
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

std::cout << "Computing " << nR_star << " " << dimMat << "x" << dimMat << " 2-center overlap matrices in the " << basis_string  << " basis..." << std::flush;

// Start the calculation
auto begin = std::chrono::high_resolution_clock::now();  

    std::vector<std::array<double,4>> overlap2Matrices;
    overlap2Matrices.reserve(total_elem);

    #pragma omp parallel for schedule(static,1) reduction(merge: overlap2Matrices)  
    for(uint64_t s = 0; s < total_elem; s++){ //Spans the lower triangle of all the nR_star matrices <P,0|P',R>
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

// Store the matrices and the list of direct lattice vectors
uint64_t n_entries = overlap2Matrices.size();
std::ofstream output_file(IntFiles_Dir + "o2Mat_" + intName + ".o2c");
output_file << "2-CENTER OVERLAP INTEGRALS" << std::endl;
output_file << "Requested nR: " << nR << ". Computed nR: " << nR_star << std::endl;
output_file << "Tolerance: 10^-" << tol << ". Matrix density: " << ((double)n_entries/total_elem)*100 << " %" << std::endl;
output_file << "Entry, mu, mu', R" << std::endl;
output_file << n_entries << std::endl;
output_file << dimMat_AUX << std::endl;
output_file.precision(12);
output_file << std::scientific;
for(uint64_t ent = 0; ent < n_entries; ent++){
    output_file << overlap2Matrices[ent][0] << "  " << static_cast<uint32_t>(overlap2Matrices[ent][1]) << " " << 
    static_cast<uint32_t>(overlap2Matrices[ent][2]) << " " << static_cast<uint32_t>(overlap2Matrices[ent][3]) << std::endl;
}
combs.save(IntFiles_Dir + "RlistFrac_" + intName + ".o2c", arma::arma_ascii);

std::cout << "Done! Elapsed wall-clock time: " << std::to_string( elapsed.count() * 1e-3 ) << " seconds." << std::endl;
std::cout << "Values above 10^-" << std::to_string(tol) << " stored in the file: " << IntFiles_Dir + "o2Mat_" + intName + ".o2c" << 
    " , and list of Bravais vectors in " << IntFiles_Dir + "RlistFrac_" + intName + ".o2c" << std::endl;

// double trace_overlap0 {arma::trace(overlap2Matrices.slice(0))};
// if(std::abs(trace_overlap0-dimMat) >= 0.1){
//     std::cerr << "UNEXPECTED ERROR! There is a deviation of " << 100*abs(trace_overlap0-dimMat)/dimMat << 
//         "% in the trace of the 2-center overlap matrix in the reference cell." << std::endl;
//     throw std::logic_error("Please, contact the developers.");
// }

}

/**
 * Method to compute the E^{i,i'}_{0} Hermite Gaussian coefficients, for i,i'<=4. Returns only the t=0 component.
 * @param index Bijection (i,i') to a single integer, given by index(i,i')= i' + i(i+1)/2.
 * @param p Sum of the exponents of the two individual Gaussians.
 * @param PA The corresponding spatial component of the vector going from the center of first Gaussian to the center of the Hermite Gaussian.
 * @param PB The corresponding spatial component of the vector going from the center of second Gaussian to the center of the Hermite Gaussian.
 * @return double E^{i,i'}_{0}.
 */
double IntegralsOverlap2C::Efunt0(const int index, const double p, const double PA, const double PB){

    switch(index)
    {
    case 0:  {// (i,j) = (0,0)
        return 1.0;
    }
    case 1:  {// (i,j) = (1,0)
        return PA;
    }
    case 2:  {// (i,j) = (1,1)
        return (PA*PB + 0.5/p);
    }
    case 3:  {// (i,j) = (2,0)
        return (PA*PA + 0.5/p);
    }
    case 4:  {// (i,j) = (2,1)
        double facp = 0.5/p;
        return (PA*2*facp + PB*(PA*PA + facp)); 
    }
    case 5:  {// (i,j) = (2,2)
        double facp = 0.5/p;
        return (facp*(3*facp + PB*(4*PA + PB)) + PA*PA*(facp + PB*PB));  
    }
    case 6:  {// (i,j) = (3,0)
        return (PA*(PA*PA*p + 1.5)/p);
    }
    case 7:  {// (i,j) = (3,1)
        double facp = 0.5/p;
        return ((PA*p*(6*(PA + PB) + PB*PA*PA*p*4) + 3)*facp*facp);
    }
    case 8:  {// (i,j) = (3,2)
        double facp = 0.5/p;
        return ((PA*p*(PA*PA*(PB*PB*p*4 + 2) + 6*PB*(2*PA + PB)) + 9*PA + 6*PB)*facp*facp);
    }
    case 9:  {// (i,j) = (3,3)
        double facp = 0.5/p;
        double PAPAp = PA*PA*p;
        double PBPBp = PB*PB*p;
        double PAPBp = PA*PB*p;
        return ((PAPBp*(PAPAp*(PBPBp*8 + 12) + 12*(PBPBp + 3*PAPBp) + 54) + 18*(PAPAp + PBPBp) + 15)*facp*facp*facp);
    }
    case 10: {// (i,j) = (4,0)
        double facp = 0.5/p;
        double PAPAp = PA*PA*p; 
        return ((PAPAp*4*(PAPAp + 3) + 3)*facp*facp);
    }
    case 11: {// (i,j) = (4,1)
        double facp = 0.5/p;
        return ((PA*PA*p*4*((PA*PB*p + 2)*PA + 3*PB) + 12*PA + 3*PB)*facp*facp); 
    }
    case 12: {// (i,j) = (4,2)
        double facp = 0.5/p;
        double PAPAp = PA*PA*p;
        return ((PAPAp*(PAPAp*(4 + PB*PB*p*8) + PB*p*(32*PA + 24*PB) + 36) + PB*6*p*(8*PA + PB) + 15)*facp*facp*facp);
    }
    case 13: {// (i,j) = (4,3)
        double facp = 0.5/p;
        double PAPAp = PA*PA*p;
        double PBPBp = PB*PB*p;
        return ((PAPAp*(PAPAp*PB*(8*PBPBp + 12) + 24*PBPBp*(2*PA + PB) + 24*PA + 108*PB) + PBPBp*(72*PA + 6*PB) + 60*PA + 45*PB)*facp*facp*facp);
    }
    case 14: {// (i,j) = (4,4)
        double facp = 0.5/p;
        double facp_to2 = facp*facp;
        double PAPAp = PA*PA*p;
        double PBPBp = PB*PB*p;
        double PAPBp = PA*PB*p;
        return ((PAPAp*(PAPAp*(PBPBp*16*(PBPBp + 3) + 12) + PAPBp*(PBPBp*128 + 192) + PBPBp*48*(PBPBp + 9) + 180)
        + PAPBp*(PBPBp*192 + 480) + PBPBp*12*(PBPBp + 15) + 105)*facp_to2*facp_to2);
    }
    default: {
        throw std::invalid_argument("IntegralsOverlap2C::Efunt0 error: the E^{i,i'}_{0} coefficients are being evaluated for i and/or i' >= 5");
    }
    }
        
}


}