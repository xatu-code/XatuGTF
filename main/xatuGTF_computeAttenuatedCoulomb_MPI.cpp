#include <mpi.h>
#include "xatu.hpp"

int main(int argc, char* argv[]){

    int procMPI_rank, procMPI_size;

    MPI_Init (&argc,&argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &procMPI_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &procMPI_size);
    if(procMPI_rank == 0){
        xatu::printHeaderGTFprovisional();
        std::cout << std::endl;
        xatu::printParallelizationMPI(procMPI_size);
        std::cout << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
        std::cout << "|                   ATTENUATED COULOMB (2C & 3C) INTEGRALS                  |" << std::endl;
        std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    }

    // INPUT PARAMETERS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string outp_file = "InputFiles/Pho_PBE0_1D_0f.outp";
    int ncells = 43;                                 //Number of H(R) and S(R) matrices taken into account from the .outp file
    std::string bases_file = "InputFiles/Bases_custom.txt";
    std::string savefile = "w050custom";        //Result will be saved in file Results/1-Integrals/o2Mat_savefile.o2c & o3Mat_savefile.o3c
    double omega = 0.5;                               //Attenuation parameter in erfc(omega*|r-r'|), in atomic units (length^-1)
    int nR = 100;                                     //Minimum number of direct lattice vectors for which the 2-center overlap integrals will be computed
    int nR2 = 100;                                    //Square root of the minimum number of direct lattice vectors for which the 3-center overlap integrals will be computed
    int tol2C = 8;                                   //Threshold tolerance for the overlap 2C integrals: only entries > 10^-tol are stored
    int tol3C = 8;                                    //Threshold tolerance for the overlap 3C integrals: only entries > 10^-tol are stored
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    xatu::ConfigurationCRYSTAL_MPI CRYSTALconfig(outp_file, procMPI_rank, procMPI_size, ncells, true);
    xatu::ConfigurationGTF_MPI GTFconfig(procMPI_rank, procMPI_size, CRYSTALconfig.nspecies, CRYSTALconfig.atomic_number_ordering, bases_file);
    xatu::IntegralsBase IntBase(CRYSTALconfig, GTFconfig); 

    xatu::IntegralsAttenuatedCoulomb2C_MPI AttCoulomb2C(IntBase, procMPI_rank, procMPI_size, omega, tol2C, nR,  savefile);
    xatu::IntegralsAttenuatedCoulomb3C_MPI AttCoulomb3C(IntBase, procMPI_rank, procMPI_size, omega, tol3C, nR2, savefile);

    MPI_Finalize();
    return 0;

}
