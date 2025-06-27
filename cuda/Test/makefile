# NVCC and settings
# NVCC := nvcc
# CXXFLAGS := -g -G -std=c++17 -rdc=true -lineinfo -lcnpy -lz -o PositionwiseFeedForwardTest

# # Files
# SRCS := ../PositionwiseFeedForward.cu ../DropOut.cu ../Linear.cu ../ReLU.cu ../Embedding.cu ../UtilKernel.cu ../Util.cu ../Tensor.cu ../TensorFunction.cu ../TensorGraph.cu ../TensorKernel.cu PositionwiseFeedForwardTest.cu
# OBJS := $(SRCS:.cu=.o)
# TARGET := PositionwiseFeedForwardTest

# all: $(TARGET)

# %.o: %.cu
# 	$(NVCC) $(CXXFLAGS) -c $< -o $@

# $(TARGET): $(OBJS)
# 	$(NVCC) $(CXXFLAGS) $^ -o $@


# clean:
# 	rm -f $(TARGET) $(OBJS)


NVCC := nvcc
CXXFLAGS := -g -G -std=c++17 -rdc=true -lineinfo -lcnpy -lz -o MultiheadAttentionTest

# Files
SRCS := ../MultiheadAttention.cu ../Softmax.cu ../DropOut.cu ../UtilKernel.cu ../Util.cu ../Tensor.cu ../TensorFunction.cu ../TensorGraph.cu ../TensorKernel.cu MultiheadAttentionTest.cu
OBJS := $(SRCS:.cu=.o)
TARGET := MultiheadAttentionTest

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $^ -o $@


clean:
	rm -f $(TARGET) $(OBJS)