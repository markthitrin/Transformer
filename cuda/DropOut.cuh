#ifndef DROP_OUT
#define DROP_OUT

#include "Header.h"
#include "Tensor.h"

class DropOut {
public:
    DropOut(
        Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient) noexcept;
    ~DropOut();

    cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);
    
	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

    Tensor input;
    Tensor output;
    Tensor outputGradient;
    Tensor inputGradient;

    Tensor mask;
};

#endif // !DROP_OUT
