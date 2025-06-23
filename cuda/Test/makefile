# NVCC and settings
NVCC := nvcc
CXXFLAGS := -g -G -std=c++17 -rdc=true -lineinfo -lcnpy -lz -o EmbeddingTest 

# Files
SRCS := ../Embedding.cu ../UtilKernel.cu ../Util.cu ../Tensor.cu ../TensorFunction.cu ../TensorGraph.cu ../TensorKernel.cu EmbeddingTest.cu
OBJS := $(SRCS:.cu=.o)
TARGET := EmbeddingTest

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $^ -o $@


clean:
	rm -f $(TARGET) $(OBJS)