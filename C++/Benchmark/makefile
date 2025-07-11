
CXX = g++
CXXFLAGS = -std=c++17 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2 -fno-omit-frame-pointer -funroll-loops -ftree-vectorize -mavx2 -msse

SRCS = ReLU.cpp ../Timer.cpp ../Embedding.cpp ../PositionalEncoder.cpp ../Transformer.cpp ../Encoder.cpp ../EncoderLayer.cpp ../Decoder.cpp ../DecoderLayer.cpp ../LayerNorm.cpp ../Linear.cpp ../ReLU.cpp ../FeedForwardBlock.cpp ../MultiheadAttention.cpp ../Softmax.cpp ../DropOut.cpp ../Tensor.cpp ../Util.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = ReLU

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)