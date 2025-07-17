
CXX = clang++
CXXFLAGS = -O3 --std=c++20 -funroll-loops -ftree-vectorize -mavx2 -msse -ffast-math -march=native -fveclib=libmvec

SRCS = \
	Embedding.cpp \
	PositionalEncoder.cpp \
	Transformer.cpp \
	Encoder.cpp \
	EncoderLayer.cpp \
	Decoder.cpp \
	DecoderLayer.cpp \
	DecoderLayer.cpp \
	LayerNorm.cpp \
	Linear.cpp \
	ReLU.cpp \
	FeedForwardBlock.cpp \
	MultiheadAttention.cpp \
	Softmax.cpp \
	DropOut.cpp \
	Tensor.cpp \
	Util.cpp \
	Data.cpp \
	Timer.cpp \
	main.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)