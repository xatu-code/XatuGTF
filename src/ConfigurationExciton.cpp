#include "xatu/ConfigurationExciton.hpp"

namespace xatu {

/**
 * File constructor for ConfigurationExciton. It extracts the relevant information from the exciton file.
 * @param exciton_file Name of file with the exciton configuration.
 */
ConfigurationExciton::ConfigurationExciton(const std::string& exciton_file) : ConfigurationBase(exciton_file){

    parseContent();

    if(mode == "tb"){
        checkContentCoherenceTB();
    }
    else if(mode == "gaussian"){
        checkContentCoherenceGTF();
    }

}

/**
 * Method to parse the exciton configuration from its file.
 * @details This method extracts all information from the configuration file and
 * stores it with the adequate format in the information struct. 
 * @return void.
 */
void ConfigurationExciton::parseContent(){

    extractArguments();
    extractRawContent();

    if (contents.empty()){
        throw std::logic_error("File contents must be extracted first");
    }

    for(const auto& arg : foundArguments){ // first determine the non-struct attributes mode, label
        auto content = contents[arg];

        if (content.size() == 0){
            continue;
        }
        else if(content.size() != 1){
            throw std::logic_error("Expected only one line per field");
        }

        if(arg == "mode"){
            std::string str = parseWord(content[0]);
            this->mode_ = str;
        }
        if(arg == "label"){
            this->label_ = standarizeLine(content[0]);
        }
    }

    if(mode == "tb"){
        for(const auto& arg : foundArguments){ // in the TB mode, distribute the parsed data in the configurationTB struct
            auto content = contents[arg];

            if(arg == "nki"){
                std::vector<int32_t> nki = parseLine<int32_t>(content[0]);
                excitonInfoTB.nki = nki;
            }
            else if(arg == "submesh"){
                excitonInfoTB.submeshFactor = parseScalar<int>(content[0]);
            }
            else if(arg == "shift"){
                std::vector<double> shift = parseLine<double>(content[0]);
                excitonInfoTB.shift = arma::colvec(shift);
            }
            else if(arg == "nbands"){
                std::vector<int> nbands = parseLine<int>(content[0]);
                excitonInfoTB.nvbands = nbands[0];
                if(nbands.size() == 2){ // flexible input: if only one value is given, it is used for both valence and conduction bands 
                    excitonInfoTB.ncbands = nbands[1];
                } else {
                    excitonInfoTB.ncbands = nbands[0];
                }
            }
            else if(arg == "bandlist"){
                // uint does not work with arma::urowvec, so we use typedef arma::uword
                std::vector<arma::s64> bands = parseLine<arma::s64>(content[0]);
                excitonInfoTB.bands = arma::ivec(bands);
            }
            else if(arg == "totalmomentum"){
                std::vector<double> Q = parseLine<double>(content[0]);
                excitonInfoTB.Q = arma::colvec(Q);
            }
            else if(arg == "cutoff"){
                excitonInfoTB.cutoff = parseScalar<double>(content[0]);
            }
            else if(arg == "dielectric"){
                std::vector<double> eps = parseLine<double>(content[0]);
                excitonInfoTB.eps = arma::vec(eps);
            }
            else if(arg == "reciprocal"){
                excitonInfoTB.methodTB = "reciprocalspace";
                excitonInfoTB.nReciprocalVectors = parseScalar<int>(content[0]);
            }
            else if(arg == "exchange"){
                std::string str = parseWord(content[0]);
                if ((str != "true") && (str != "false")){
                    throw std::invalid_argument("Exchange option must be set to 'true' or 'false'.");
                }
                if (str == "true"){
                    excitonInfoTB.exchange = true;
                }
            }
            else if(arg == "exchange.potential"){
                std::string str = parseWord(content[0]);
                std::cout << arg << " " << str << std::endl;
                excitonInfoTB.exchangePotential = str;
            }
            else if(arg == "potential"){
                std::string str = parseWord(content[0]);
                excitonInfoTB.potential = str;
            }
            else if(arg == "scissor"){
                excitonInfoTB.scissor = parseScalar<double>(content[0]);
            }
            else if(arg == "regularization"){
                excitonInfoTB.regularization = parseScalar<double>(content[0]);
            }
            else if(arg == "mode" || arg == "label"){
            }
            else{    
                std::cout << "Unexpected argument: " << arg << ", skipping block..." << std::endl;
            }
        }
    }
    else if(mode == "gaussian"){
        for(const auto& arg : foundArguments){ // in the GTF mode, distribute the parsed data in the configurationGTF struct
            auto content = contents[arg];

            if(arg == "nki"){
                std::vector<int32_t> nki = parseLine<int32_t>(content[0]);
                excitonInfoGTF.nki = nki;
            }
            else if(arg == "nbands"){
                std::vector<int> nbands = parseLine<int>(content[0]);
                excitonInfoGTF.nvbands = nbands[0];
                if(nbands.size() == 2){
                    excitonInfoGTF.ncbands = nbands[1];
                } else {
                    excitonInfoGTF.ncbands = nbands[0];
                }
            }
            else if(arg == "bandlist"){
                std::vector<arma::s64> bands = parseLine<arma::s64>(content[0]);
                excitonInfoGTF.bands = arma::ivec(bands);
            }
            else if(arg == "nkipol"){
                std::vector<int32_t> nkiPol = parseLine<int32_t>(content[0]);
                excitonInfoGTF.nkiPol = nkiPol;
            }
            else if(arg == "alpha"){
                excitonInfoGTF.alpha = parseScalar<double>(content[0]);
            }
            else if(arg == "totalmomentum"){
                std::vector<double> Q = parseLine<double>(content[0]);
                excitonInfoGTF.Q = arma::colvec(Q);
            }
            else if(arg == "scissor"){
                excitonInfoGTF.scissor = parseScalar<double>(content[0]);
            }
            else if(arg == "mode" || arg == "label"){
            }
            else{    
                std::cout << "Unexpected argument: " << arg << ", skipping block..." << std::endl;
            }
        }
    }
    else{
        throw std::logic_error("'mode' must be either 'TB' or 'GAUSSIAN'");
    }

}

/**
 * Method to check whether the information extracted from the configuration file in the TB mode is
 * consistent and well-defined. 
 * @return void.
 */
void ConfigurationExciton::checkContentCoherenceTB(){

    if(excitonInfoTB.nki.empty()){
        throw std::logic_error("'nki' must be speficied");
    }
    if(excitonInfoTB.nki.size() > 3){
        throw std::logic_error("'nki' cannot have more than 3 components");
    }
    for(auto n : excitonInfoTB.nki){
        if(n < 0){
            throw std::logic_error("'nki' must have non-negative components");
        }
    }
    if(excitonInfoTB.bands.empty() && excitonInfoTB.nvbands == 0){
        throw std::logic_error("either 'bandlist' or 'nbands' must be specified");
    }
    if(!excitonInfoTB.bands.empty() && excitonInfoTB.nvbands != 0){
        throw std::logic_error("only one of 'bandlist' and 'nbands' can be specified");
    }
    if(arma::all(excitonInfoTB.bands <= 0) || arma::all(excitonInfoTB.bands > 0)){
        throw std::logic_error("'bands' must contain both valence (<= 0) and conduction (> 0) bands");
    }
    if(excitonInfoTB.eps.empty()){
        throw std::logic_error("'dielectric' must be specified");
    }
    if(excitonInfoTB.Q.n_elem != 3){
        throw std::logic_error("'Q' must have 3 components");
    }

    bool potentialFound = false;
    bool exchangePotentialFound = false;
    for (auto potential : supportedPotentialsTB){
        if(excitonInfoTB.potential == potential){
            potentialFound = true;
        }
        if(excitonInfoTB.exchange && excitonInfoTB.exchangePotential == potential){
            exchangePotentialFound = true;
        }
    }
    if (!potentialFound){
        throw std::invalid_argument("Specified 'potential' not supported. Use 'keldysh' or 'coulomb'");
    }
    if (excitonInfoTB.exchange && !exchangePotentialFound){
        throw std::invalid_argument("Specified 'exchange.potential' not supported. Use 'keldysh' or 'coulomb'");
    }
    if (excitonInfoTB.methodTB != "realspace" && excitonInfoTB.methodTB != "reciprocalspace"){
        throw std::invalid_argument("Invalid method for potential matrix elements in TB mode. Use 'realspace' or 'reciprocalspace'");
    }

}

/**
 * Method to check whether the information extracted from the configuration file in the GAUSSIAN mode is
 * consistent and well-defined. 
 * @return void.
 */
void ConfigurationExciton::checkContentCoherenceGTF(){

    if(excitonInfoGTF.nki.empty()){
        throw std::logic_error("'nki' must be speficied");
    }
    if(excitonInfoGTF.nki.size() > 3){
        throw std::logic_error("'nki' cannot have more than 3 components");
    }
    for(auto n : excitonInfoGTF.nki){
        if(n < 0){
            throw std::logic_error("'nki' must have non-negative components");
        }
    }
    if(excitonInfoGTF.nkiPol.empty()){
        throw std::logic_error("'nkiPol' must be speficied");
    }
    if(excitonInfoGTF.nkiPol.size() > 3){
        throw std::logic_error("'nkiPol' cannot have more than 3 components");
    }
    for(auto n : excitonInfoGTF.nkiPol){
        if(n < 0){
            throw std::logic_error("'nkiPol' must have non-negative components");
        }
    }
    if(excitonInfoGTF.bands.empty() && excitonInfoGTF.nvbands == 0){
        throw std::logic_error("either 'bandlist' or 'nbands' must be specified");
    }
    if(!excitonInfoGTF.bands.empty() && excitonInfoGTF.nvbands != 0){
        throw std::logic_error("only one of 'bandlist' and 'nbands' can be specified");
    }
    if(arma::all(excitonInfoGTF.bands <= 0) || arma::all(excitonInfoGTF.bands > 0)){
        throw std::logic_error("'bands' must contain both valence (<= 0) and conduction (> 0) bands");
    }
    if(excitonInfoGTF.Q.n_elem != 3){
        throw std::logic_error("'Q' must have 3 components");
    }
    if(excitonInfoGTF.alpha < 0){
        throw std::logic_error("'alpha' must be non-negative");
    }

}

}