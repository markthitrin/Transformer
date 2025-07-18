CXX = clang++
CXXFLAGS = -std=c++17 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O3 -fno-omit-frame-pointer -funroll-loops -ftree-vectorize -mavx2 -msse

SRCS = MultiheadAttentionBackward.cpp ../Timer.cpp ../Embedding.cpp ../PositionalEncoder.cpp ../Transformer.cpp ../Encoder.cpp ../EncoderLayer.cpp ../Decoder.cpp ../DecoderLayer.cpp ../LayerNorm.cpp ../Linear.cpp ../ReLU.cpp ../FeedForwardBlock.cpp ../MultiheadAttention.cpp ../Softmax.cpp ../DropOut.cpp ../Tensor.cpp ../Util.cpp

TARGET = MultiheadAttentionBackward

all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)