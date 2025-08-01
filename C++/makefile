
CXX = clang++
CXXFLAGS = -O3 --std=c++20 -fopenmp -funroll-loops -ftree-vectorize -mavx2 -msse -ffast-math -march=native -fveclib=libmvec

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
	Util.cpp \
	Data.cpp \
	Timer.cpp \
	main.cpp

OBJDIR = build
OBJS = $(SRCS:.cpp=.o)
OBJS := $(addprefix $(OBJDIR)/,$(OBJS))

TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)