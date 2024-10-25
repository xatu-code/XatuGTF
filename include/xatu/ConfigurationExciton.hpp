#pragma once
#include <armadillo>
#include "xatu/ConfigurationBase.hpp"


namespace xatu {

/**
 * The ConfigurationExciton class is a specialization of ConfigurationBase to parse exciton configuration files.
 * It is used in both TB and GAUSSIAN modes, and the data is stored in a different struct for each of them
 */
class ConfigurationExciton : public ConfigurationBase{

    protected:
        // Simulation label
        std::string label_;
        // Mode: TB or GAUSSIAN
        std::string mode_;
    public: 
        const std::string& label = label_;
        const std::string& mode  = mode_;

    struct configurationTB {
        // Vector with the number of k-points in each Gi direction for the BSE. Only the first ndim components will be taken into account
        std::vector<int32_t> nki;
        // Number of valence and conduction single-particle bands included in the exciton basis
        int nvbands = 0;
        int ncbands = 0;
        // Reduction factor of the BZ mesh. Defaults to 1
        int submeshFactor = 1;
        // List of bands included in the exciton basis, where 0 is the highest valence band (conduction >=1, valence <=0)
        arma::ivec bands = {};
        // Center-of-mass momentum of the exciton. It must always have 3 components
        arma::colvec Q = {0., 0., 0.};
        // Displacement vector of the center of the BZ mesh
        arma::colvec shift;
        // Cutoff to be used 
        double cutoff;
        // Dielectric constants
        arma::vec eps = {};
        // Screening length
        double r0;
        // Thickness of layer
        double d;
        // Calculation method for the potential matrix elements in TB mode (either 'realspace' or 'reciprocalspace')
        std::string methodTB = "realspace";
        // Flag to compute the exciton spectrum with exchange
        bool exchange = false;
        // Scissor cut to correct the bandgap, given in eV
        double scissor = 0.0;
        // Number of reciprocal vectors to use in reciprocal space calculation
        int nReciprocalVectors = 0;
        // Potential to use in direct term
        std::string potential = "keldysh";
        // Potential to use in exchange if active
        std::string exchangePotential = "keldysh";
        // Regularization distance
        double regularization = 0.0;
    };

    struct configurationGTF {
        // Vector with the number of k-points in each Gi direction for the BSE. Only the first ndim components will be taken into account
        std::vector<int32_t> nki;
        // Number of valence and conduction single-particle bands included in the exciton basis
        int nvbands = 0;
        int ncbands = 0;
        // List of bands included in the exciton basis, where 0 is the highest valence band (conduction >=1, valence <=0)
        arma::ivec bands = {};
        // Vector with the number of k-points in each Gi direction for the auxiliary polarizability matrix Pi. Only the first ndim components 
        // will be taken into account
        std::vector<int32_t> nkiPol;
        // Regularization parameter to invert the 2-center metric matrix in the AUX basis
        double alpha = 0.0;
        // Center-of-mass momentum of the exciton, in fractional coordinates. It must always have 3 components
        arma::colvec Q = {0., 0., 0.};
        // Scissor cut to correct the bandgap, given in eV
        double scissor = 0.0;
    };

    public:
        configurationTB excitonInfoTB;
        configurationGTF excitonInfoGTF;
        std::vector<std::string> supportedPotentialsTB = {"keldysh", "coulomb"};
    
    protected:
        ConfigurationExciton() = default;
    public:
        ConfigurationExciton(const std::string& exciton_file);
    
    protected:
        void parseContent() override;
        void checkContentCoherenceTB();
        void checkContentCoherenceGTF();
};

}