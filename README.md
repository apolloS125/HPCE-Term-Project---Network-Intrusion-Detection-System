# HPCE Term Project - Network Intrusion Detection System
## SVM + DBSCAN with 5 Parallel Processing Techniques

### Project Structure
```
hpce_code/
├── Makefile                    # Build all techniques
├── README.md                   # This file
├── scripts/
│   └── generate_data.py        # Generate synthetic KDD-like dataset
├── src/
│   ├── common.h                # Shared: SVM, DBSCAN, I/O, timing
│   ├── sequential.cpp          # Sequential baseline
│   ├── cpp_thread.cpp          # C++ std::thread parallel
│   ├── openmp.cpp              # OpenMP parallel
│   ├── mpi_ids.cpp             # MPI distributed
│   ├── cuda_ids.cu             # CUDA GPU accelerated
│   └── pyspark_ids.py          # PySpark distributed
├── data/                       # Generated dataset (created by generate_data.py)
└── bin/                        # Compiled binaries
```

### Quick Start
```bash
# Generate data + build + run (Sequential, Thread, OpenMP, PySpark)
make all
make run

# Build individual techniques
make sequential
make thread
make openmp
make mpi          # requires mpicxx
make cuda         # requires nvcc

# Run individual techniques
./bin/sequential
./bin/cpp_thread 8           # 8 threads
./bin/openmp 8               # 8 threads
mpirun -np 4 ./bin/mpi_ids   # 4 processes
./bin/cuda_ids               # requires NVIDIA GPU
python3 src/pyspark_ids.py 4 # 4 Spark workers
```

### Dataset Options
```bash
# Default: 10,000 samples, 34 features
python3 scripts/generate_data.py

# Custom size (for benchmarking)
python3 scripts/generate_data.py 50000 34    # 50K samples
python3 scripts/generate_data.py 100000 41   # 100K samples, 41 features (KDD)
```

### Pipeline: SVM-First, DBSCAN-Second
1. **Preprocessing**: Load CSV, normalize features
2. **SVM Classification**: Train RBF kernel SVM, predict test data with confidence scores
3. **DBSCAN Anomaly Detection**: Process only uncertain subset (confidence < threshold)
4. **Output**: Known attack labels + unknown attack alerts + performance metrics

### Performance Metrics
Each technique reports:
- **Execution time** (ms) with breakdown per stage
- **FLOP count** (total floating point operations)
- **GFLOPS** (throughput)
- **Accuracy** and confusion matrix
- **Unknown attack count** (DBSCAN noise points)

### Dependencies
| Technique | Compiler/Tool | Flag |
|-----------|--------------|------|
| Sequential | g++ | -O2 |
| C++ Thread | g++ | -pthread |
| OpenMP | g++ | -fopenmp |
| MPI | mpicxx | - |
| CUDA | nvcc | - |
| PySpark | python3 | pip install pyspark |

### Monitoring CPU Usage
```bash
# In another terminal while running:
top
# Press '1' to see per-core usage
```
