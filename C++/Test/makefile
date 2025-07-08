
CXX = clang++
CXXFLAGS = -g -lcnpy -lz --std=c++20

SRCS = TransformerTest.cpp ../Embedding.cpp ../PositionalEncoder.cpp ../Transformer.cpp ../Encoder.cpp ../EncoderLayer.cpp ../Decoder.cpp ../DecoderLayer.cpp ../DecoderLayer.cpp ../LayerNorm.cpp ../Linear.cpp ../ReLU.cpp ../FeedForwardBlock.cpp ../MultiheadAttention.cpp ../Softmax.cpp ../DropOut.cpp ../Tensor.cpp ../Util.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = TransformerTest

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)