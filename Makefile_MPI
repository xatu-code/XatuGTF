# Compiler & compiler flags
CC = mpic++
FC = gfortran
CFLAGS += -Wall -std=c++17
FFLAGS := -Wall -Wno-tabs 

# MKL directory, relevant only with the flag MKL=1
MKL_DIR = /mnt/home/soft/intel2023/intel/oneapi/mkl/2023.1.0

# Include folders
INCLUDE := -I$(PWD)/include

# Libraries
LIBS := -DARMA_DONT_USE_WRAPPER -DARMA_64BIT_WORD -DARMA_BLAS_LONG_LONG -lstdc++ -L$(PWD) -lxatu -lm -lopenblas -llapack -fopenmp -lgfortran

# Conditional flags for compilation
ifeq ($(DEBUG), 1)
	CFLAGS := -g $(CFLAGS)
	FFLAGS := -g $(FFLAGS)
else 
	CFLAGS := -O2 $(CFLAGS)
	FFLAGS := -O2 $(FFLAGS)
endif
ifeq ($(HDF5), 1)
	CFLAGS += -DARMA_USE_HDF5
	LIBS += -lhdf5
endif
ifeq ($(MKL), 1)
	INCLUDE := -I$(MKL_DIR)/include $(INCLUDE)
	LIBS := -L$(MKL_DIR)/lib/intel64 -DMKL_ILP64 -m64 -lmkl_rt -lmkl_blas95_lp64 -lmkl_lapack95_lp64 -lm $(LIBS)
endif
ifeq ($(NO_DEBUG), 1)
	LIBS := -DARMA_NO_DEBUG $(LIBS)
endif

# Compilation targets
CC_SRC_FILES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp, build/%.o, $(CC_SRC_FILES))
FC_SRC_FILES := $(wildcard src/*.f90)
OBJECTS_FC := $(patsubst src/%.f90, build/%.o, $(FC_SRC_FILES))
OBJECTS += $(OBJECTS_FC)

# Create folders
dummy_build_folder := $(shell mkdir -p build)
dummy_bin_folder := $(shell mkdir -p bin)

build:	$(OBJECTS)
	ar rcs libxatu.a $(OBJECTS) 
	
xatu: main/xatu.cpp $(OBJECTS) 
	$(CC) -o bin/$@ $< $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LIBS)

%: main/%.cpp $(OBJECTS)
	$(CC) -o bin/$@ $< $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LIBS)

# Compilation steps
# $< refers to first prerequisite and $@ to the target
build/%.o: src/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS) $(LDFLAGS) $(INCLUDE) $(LIBS) 

build/%.o: src/%.f90
	$(FC) -c $< -o $@ $(LDFLAGS) $(FFLAGS) $(INCLUDE) $(LIBS)

clean:
	rm -f build/*.o bin/* libxatu.a
