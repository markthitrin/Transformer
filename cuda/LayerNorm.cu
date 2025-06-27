#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "LayerNorm.cuh"

LayerNorm::LayerNorm(
	Tensor& input,
	Tensor& output,
	Tensor& outputGradient,
	Tensor& inputGradient) noexcept:

	input(input),
	output(output),
	outputGradient(outputGradient),
	inputGradient(inputGradient),
	
	alpha(1, dModel),
	bias(1, dModel),

	alphaOpt(1, dModel),
	biasOpt(1, dModel),

	mean(1, batch * sequenceLength),
	std(1, batch * sequenceLength),
	xHat(batch * sequenceLength, dModel),

    sumG(1, batch * sequenceLength),
    sumGXhat(1, batch * sequenceLength) {
	
	Set(alpha, 1.0f);
	Reset(bias);
}
LayerNorm::~LayerNorm() {
	alpha.free();
	bias.free();
	xHat.free();
	std.free();
}

cudaGraphNode_t LayerNorm::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = AppendMeanNode(graph, dependencyNodes, input, mean);
	cudaGraphNode_t k2 = AppendStdNode(graph, {k1}, input, mean, std);
	cudaGraphNode_t k3 = AppendLayerNormNode(graph, {k2}, input, mean, std, alpha, bias, xHat, output);
	return k3;
}

cudaGraphNode_t LayerNorm::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t LayerNorm::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendPlusReduceInplceBatchNode(graph, dependencyNodes, biasOpt.gradient, outputGradient, outputGradient.row);
    cudaGraphNode_t k2 = AppendPlusProductReduceInplceBatchNode(graph, {k1}, alphaOpt.gradient, outputGradient, xHat, batch * sequenceLength);
	cudaGraphNode_t k3 = AppendReduceSumNode(graph, {k2}, outputGradient, sumG);
	cudaGraphNode_t k4 = AppendReduceSumOfProductNode(graph, {k3}, outputGradient, xHat, sumGXhat);
	cudaGraphNode_t k5 = AppendLayerNormBackwardNode(graph, {k4}, outputGradient, xHat, std, sumG, sumGXhat, alpha, inputGradient);
    cudaGraphNode_t k6 = SyncDependency(graph, {k5});
	return k6;
}

cudaGraphNode_t LayerNorm::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = AppendAdamOptNode(graph, {k1}, alpha, alphaOpt);
    cudaGraphNode_t k3 = AppendAdamOptNode(graph, {k1}, bias, biasOpt);
    cudaGraphNode_t k4 = SyncDependency(graph, {k2, k3});
    return k4;
}

void LayerNorm::loadParam(cnpy::npz_t npFile, std::string prefix) {
	alpha.loadNp(npFile, prefix + ".alpha");
	bias.loadNp(npFile, prefix + ".bias");
}

void LayerNorm::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);

    target.loadNp(npFile, prefix + ".output");
    input.loadNp(npFile, prefix + ".input");

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void LayerNorm::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
	Tensor updatedAlpha(1, dModel);
	Tensor updatedBias(1, dModel);
    updatedAlpha.loadNp(npFile, prefix + ".updated_alpha");
    updatedBias.loadNp(npFile, prefix + ".updated_bias");

    PrintTestResult("backward " + prefix + ".updated_alpha", alpha, updatedAlpha);
	PrintTestResult("backward " + prefix + ".updated_bias", bias, updatedBias);
}

void LayerNorm::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    cudaDeviceSynchronize();

    // load input
    input.loadNp(npFile, prefix + ".input");

    // Forward, backward, update
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    
    this->AppendGraphUpdateParameter(graph, {k2});
    cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();
    
    checkUpdatedParam(npFile, prefix);
}

cudaGraphNode_t AppendLayerNormNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& mean, Tensor& std, Tensor& alpha, Tensor& bias, Tensor& xHat, Tensor& output) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

	const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(input.col, BLOCK_SIZE), ceil(input.row, BLOCK_SIZE));

    void* kernelArgs[] = {
		&input.data, &mean.data, &std.data, &alpha.data, &bias.data, &xHat.data, &output.data, 
		&input.pitch, &xHat.pitch, &output.pitch,
		&input.row, &input.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)LayerNormKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendLayerNormBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& outputGradient, Tensor& xHat, Tensor& std, Tensor& sumG, Tensor& sumGXhat, Tensor& alpha, Tensor& inputGradient) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

	const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCK_SIZE), ceil(outputGradient.row, BLOCK_SIZE));

    void* kernelArgs[] = {
		&outputGradient.data, &xHat.data, &std.data, &sumG.data, &sumGXhat.data, &alpha.data, &inputGradient.data,
		&outputGradient.pitch, &xHat.pitch, &inputGradient.pitch,
		&outputGradient.row, &outputGradient.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)LayerNormBackwardKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

__global__ void LayerNormKernel(
	const float* A, const float* mean, const float* std, const float* alpha, const float* bias, float* C, float* D,
	const std::size_t pitchA, const std::size_t pitchC, const std::size_t pitchD,
	const std::size_t row, const std::size_t col) {
		
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = (*Get(A, r, c, pitchA) - *Get(mean, 0, r, 0)) / (*Get(std, 0, r, 0) + eps);
		*Get(D, r, c, pitchD) = *Get(alpha, 0, c, 0) * *Get(C, r, c, pitchC) + *Get(bias, 0, c, 0);
	}
}
__global__ void LayerNormBackwardKernel(
	const float* A, const float* B, const float* std, const float* sumG, const float* sumGXHat, const float* alpha, float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		float a = __frcp_rn(col) * *Get(sumG, 0, r, 0);
		float b = __frcp_rn(col) * *Get(sumGXHat, 0, r, 0);
		*Get(C, r, c, pitchC) = __frcp_rn(*Get(std, 0, r, 0) + eps) * 
								(*Get(A, r, c, pitchA) - a - *Get(B, r, c, pitchB) * b) *
								*Get(alpha, 0, c, 0);
	}
}

