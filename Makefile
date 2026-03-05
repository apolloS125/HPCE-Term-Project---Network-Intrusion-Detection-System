# Makefile for HPCE Network IDS Project
# SVM + DBSCAN with 5 parallel techniques

CXX = g++
MPICXX = mpicxx
NVCC = nvcc
CXXFLAGS = -O2 -std=c++20
LDFLAGS = -lm

SRC = src
BIN = bin

.PHONY: all clean data sequential thread openmp mpi cuda pyspark run

all: data sequential thread openmp

# Create output directory
$(BIN):
	mkdir -p $(BIN)

# Generate dataset
data:
	cd $(SRC)/.. && python3 scripts/generate_data.py 10000 34

# Sequential baseline
sequential: $(BIN)
	$(CXX) $(CXXFLAGS) -o $(BIN)/sequential $(SRC)/sequential.cpp $(LDFLAGS)
	@echo "Built: sequential"

# C++ Thread
thread: $(BIN)
	$(CXX) $(CXXFLAGS) -pthread -o $(BIN)/cpp_thread $(SRC)/cpp_thread.cpp $(LDFLAGS)
	@echo "Built: cpp_thread"

# OpenMP
openmp: $(BIN)
	$(CXX) $(CXXFLAGS) -fopenmp -o $(BIN)/openmp $(SRC)/openmp.cpp $(LDFLAGS)
	@echo "Built: openmp"

# MPI (requires mpicxx)
mpi: $(BIN)
	$(MPICXX) $(CXXFLAGS) -o $(BIN)/mpi_ids $(SRC)/mpi_ids.cpp $(LDFLAGS)
	@echo "Built: mpi_ids"

# CUDA (requires nvcc)
cuda: $(BIN)
	$(NVCC) -O2 -o $(BIN)/cuda_ids $(SRC)/cuda_ids.cu $(LDFLAGS)
	@echo "Built: cuda_ids"

# Run all
run: data
	@echo "\n\n============ SEQUENTIAL ============"
	cd $(SRC)/.. && $(BIN)/sequential
	@echo "\n\n============ C++ THREAD ============"
	cd $(SRC)/.. && $(BIN)/cpp_thread 8
	@echo "\n\n============ OPENMP ============"
	cd $(SRC)/.. && $(BIN)/openmp 8
	@echo "\n\n============ PYSPARK ============"
	cd $(SRC)/.. && python3 $(SRC)/pyspark_ids.py 4

# Run MPI separately (needs mpirun)
run-mpi:
	cd $(SRC)/.. && mpirun -np 4 $(BIN)/mpi_ids

# Run CUDA separately (needs GPU)
run-cuda:
	cd $(SRC)/.. && $(BIN)/cuda_ids

clean:
	rm -rf $(BIN) data
