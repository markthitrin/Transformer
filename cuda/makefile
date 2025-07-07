NVCC := nvcc
NVCCFLAGS := -g -rdc=true -std=c++17

SRCS := \
	Data.cu \
	Decoder.cu \
	DecoderLayer.cu \
	DropOut.cu \
	Embedding.cu \
	Encoder.cu \
	EncoderLayer.cu \
	LayerNorm.cu \
	Linear.cu \
	main.cu \
	MultiheadAttention.cu \
	PositionalEncoder.cu \
	PositionwiseFeedForward.cu \
	ReLU.cu \
	Softmax.cu \
	Tensor.cu \
	TensorFunction.cu \
	TensorGraph.cu \
	TensorKernel.cu \
	Transformer.cu \
	Util.cu \
	UtilKernel.cu

OBJS := $(patsubst %.cu,build/%.o,$(SRCS))

TARGET := BitcoinMiner

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

build/%.o: %.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) build/*.o