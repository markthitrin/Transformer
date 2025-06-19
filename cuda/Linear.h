#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.cuh"
#include "Util.cuh"

class Linear {
public:
	Linear(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient,
		const std::size_t in,
		const std::size_t out) noexcept:

		input(input),
		output(output),
		outputGradient(outputGradient),
		inputGradient(inputGradient),
		weight(in, out),
		bias(1, out),
		weightOpt(in, out),
		biasOpt(1, out) {

		weight.HeNormalFill();
		bias.HeNormalFill();
	}
	~Linear() {
		weight.free();
		bias.free();
	}

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
		cudaGraphNode_t k1 = AppendCopyBatchNode(graph, dependencyNodes, bias, input, batch);
		cudaGraphNode_t k2 = AppendMatMulPlus(graph, {k1}, input, weight, output, false, false);
		return k2;
	}

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
		return AppendGraphForward(graph, dependencyNodes);
	}

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
		
	}

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
		
	}

	void backward() noexcept {
		feedCount++;
		Reset(_inputGradient);
		for (int i = 0; i < row; i++) {
			Plus(_biasOpt.gradient, _outputGradient.template sliceRow<1>(i), _biasOpt.gradient);
		}

		MatMulPlusATB(_outputGradient, _input, _weightOpt.gradient);
		MatMulPlusAB(_outputGradient, _weight, _inputGradient);
	}

	void updateParameter() noexcept {
		AdamOpt(_weight, _weightOpt, feedCount);
		AdamOpt(_bias, _biasOpt, feedCount);
		feedCount = 0;
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor<tgtVocab, dModel> weightUpdated;
		Tensor<1, tgtVocab> biasUpdated;
		weightUpdated.loadNp(npFile, prefix + ".updated_weight");
		biasUpdated.loadNp(npFile, prefix + ".updated_bias");

		PrintTestResult("backward " + prefix + ".weight", _weight, weightUpdated);
		PrintTestResult("backward " + prefix + ".bias", _bias, biasUpdated);
	}

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor weight;
	Tensor bias;

	int feedCount = 0;
	AdamOptimizer weightOpt;
	AdamOptimizer biasOpt;
};

#endif // !LINEAR
