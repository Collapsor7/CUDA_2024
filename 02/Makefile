# Compiler settings
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_60

# Output binary
TARGET = cuda

# Source files
GPU_SRC = adi3d_cuda.cu

# Build rules
all: build

cpu: $(CPU_SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(GPU_SRC)

gpu: $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(GPU_SRC)

clean:
	rm -f $(TARGET)

run_cpu:
	./$(TARGET) cpu

run_gpu:
	./$(TARGET) gpu

