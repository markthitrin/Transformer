#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "Softmax.cuh"
#include "DropOut.cuh"
#include "MultiheadAttention.cuh"

MultiheadAttention::MultiheadAttention(
	Tensor& inputQ,
	Tensor& inputK,
	Tensor& inputV,
	Tensor& output,
	Tensor& outputGradient,
	Tensor& inputGradientQ,
	Tensor& inputGradientK,
	Tensor& inputGradientV,
	MaskType maskType,
	std::size_t*& seqH) noexcept:

	softmax(A, As, AsGradient, AGradient, batch * head * sequenceLength, sequenceLength),
	dropout(As, Ad, AdGradient, AsGradient, batch * head * sequenceLength, sequenceLength),

	inputQ(inputQ),
	inputK(inputK),
	inputV(inputV),
	output(output),
	outputGradient(outputGradient),
	inputGradientQ(inputGradientQ),
	inputGradientK(inputGradientK),
	inputGradientV(inputGradientV),
	maskType(maskType),
	seqH(seqH),
	
	WQ(dModel, dModel),
	WK(dModel, dModel),
	WV(dModel, dModel),
	WO(dModel, dModel),

	WQOpt(dModel, dModel),
	WKOpt(dModel, dModel),
	WVOpt(dModel, dModel),
	WOOpt(dModel, dModel),

	QT(batch * dModel, sequenceLength),
	KT(batch * dModel, sequenceLength),
	VT(batch * dModel, sequenceLength),
	A(batch * head * sequenceLength, sequenceLength),
	As(batch * head * sequenceLength, sequenceLength),
	Ad(batch * head * sequenceLength, sequenceLength),
	OT(batch * dModel, sequenceLength),

	QTGradient(batch * dModel, sequenceLength),
	KTGradient(batch * dModel, sequenceLength),
	VTGradient(batch * dModel, sequenceLength),
	AGradient(batch * head * sequenceLength, sequenceLength),
	AsGradient(batch * head * sequenceLength, sequenceLength),
	AdGradient(batch * head * sequenceLength, sequenceLength),
	OTGradient(batch * dModel, sequenceLength),
	
	forwardMaskNode(nullptr),
	forwardMaskParams({}),
	backwardMaskNode(nullptr),
	backwardMaskParams({}) {

	WQ.XavierUniformFill();
	WK.XavierUniformFill();
	WV.XavierUniformFill();
	WO.XavierUniformFill();

	cudaMalloc(&seq, sizeof(std::size_t) * batch * head);
}
MultiheadAttention::~MultiheadAttention() {
	cudaFree(seq);
}

void MultiheadAttention::UpdateGraph() {
	std::size_t buffer[batch][head];
	for(int i = 0;i < batch;i++) {
		for(int j = 0;j < head;j++) {
			buffer[i][j] = seqH[i];
		}
	}
	cudaMemcpy(seq, buffer, sizeof(std::size_t) * batch * head,  cudaMemcpyHostToDevice);
}

cudaGraphNode_t MultiheadAttention::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
	cudaGraphNode_t k2 = AppendResetNode(graph, { k1 }, QT);
	cudaGraphNode_t k3 = AppendResetNode(graph, { k1 }, KT);
	cudaGraphNode_t k4 = AppendResetNode(graph, { k1 }, VT);
	cudaGraphNode_t k5 = AppendResetNode(graph, { k1 }, A);
	cudaGraphNode_t k6 = AppendResetNode(graph, { k1 }, As);
	cudaGraphNode_t k7 = AppendResetNode(graph, { k1 }, Ad);
	cudaGraphNode_t k8 = AppendResetNode(graph, { k1 }, OT);
	cudaGraphNode_t k9 = AppendResetNode(graph, { k1 }, output);

	cudaGraphNode_t k10 = AppendMatMulPlusBatchNode(graph, { k2 }, WQ, inputQ, QT, false, true, batch, true, false, false);
	cudaGraphNode_t k11 = AppendMatMulPlusBatchNode(graph, { k3 }, WK, inputK, KT, false, true, batch, true, false, false);
	cudaGraphNode_t k12 = AppendMatMulPlusBatchNode(graph, { k4 }, WV, inputV, VT, false, true, batch, true, false, false);

	cudaGraphNode_t k13 = AppendMatMulPlusBatchNode(graph, { k5, k10, k11 }, QT, KT, A, true, false, batch * head, false, false, false);
	cudaGraphNode_t k14 = AppendDivInplaceNode(graph, { k13 }, A, std::sqrt(dModel / head));
	cudaGraphNode_t k15;

	switch(maskType) {
		case LOOK_AHEAD : k15 = AppendLookAheadMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
		case PADDING: k15 = AppendPaddingMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
		case CROSS_PADDING: k15 = AppendCrossPaddingMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
	}


	cudaGraphNode_t k16 = softmax.AppendGraphForward(graph, { k6, k15 });
	cudaGraphNode_t k17 = dropout.AppendGraphForward(graph, { k7, k16 });
	cudaGraphNode_t k18 = AppendMatMulPlusBatchNode(graph, { k8, k12, k17 }, VT, Ad, OT, false, true, batch * head, false, false, false);
	cudaGraphNode_t k19 = AppendMatMulPlusBatchNode(graph, { k9, k18 }, OT, WO, output, true, false, batch, false, true, false);
	return k19;
}

cudaGraphNode_t MultiheadAttention::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
	cudaGraphNode_t k2 = AppendResetNode(graph, { k1 }, QT);
	cudaGraphNode_t k3 = AppendResetNode(graph, { k1 }, KT);
	cudaGraphNode_t k4 = AppendResetNode(graph, { k1 }, VT);
	cudaGraphNode_t k5 = AppendResetNode(graph, { k1 }, A);
	cudaGraphNode_t k6 = AppendResetNode(graph, { k1 }, As);
	cudaGraphNode_t k7 = AppendResetNode(graph, { k1 }, Ad);
	cudaGraphNode_t k8 = AppendResetNode(graph, { k1 }, OT);
	cudaGraphNode_t k9 = AppendResetNode(graph, { k1 }, output);

	cudaGraphNode_t k10 = AppendMatMulPlusBatchNode(graph, { k2 }, WQ, inputQ, QT, false, true, batch, true, false, false);
	cudaGraphNode_t k11 = AppendMatMulPlusBatchNode(graph, { k3 }, WK, inputK, KT, false, true, batch, true, false, false);
	cudaGraphNode_t k12 = AppendMatMulPlusBatchNode(graph, { k4 }, WV, inputV, VT, false, true, batch, true, false, false);

	cudaGraphNode_t k13 = AppendMatMulPlusBatchNode(graph, { k5, k10, k11 }, QT, KT, A, true, false, batch * head, false, false, false);
	cudaGraphNode_t k14 = AppendDivInplaceNode(graph, { k13 }, A, std::sqrt(dModel / head));
	cudaGraphNode_t k15;
	switch(maskType) {
		case LOOK_AHEAD : k15 = AppendLookAheadMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
		case PADDING: k15 = AppendPaddingMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
		case CROSS_PADDING: k15 = AppendCrossPaddingMaskBatchNode(graph, { k14 }, A, batch * head, seq, -1e9); break;
	}

	cudaGraphNode_t k16 = softmax.AppendGraphForward(graph, { k6, k15 });
	cudaGraphNode_t k17 = dropout.AppendGraphPredict(graph, { k7, k16 });
	cudaGraphNode_t k18 = AppendMatMulPlusBatchNode(graph, { k8, k12, k17 }, VT, Ad, OT, false, true, batch * head, false, false, false);
	cudaGraphNode_t k19 = AppendMatMulPlusBatchNode(graph, { k9, k18 }, OT, WO, output, true, false, batch, false, true, false);
	return k19;
}

cudaGraphNode_t MultiheadAttention::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
	cudaGraphNode_t k2 = AppendResetNode(graph, { k1 }, QTGradient);
	cudaGraphNode_t k3 = AppendResetNode(graph, { k1 }, KTGradient);
	cudaGraphNode_t k4 = AppendResetNode(graph, { k1 }, VTGradient);
	cudaGraphNode_t k5 = AppendResetNode(graph, { k1 }, AGradient);
	cudaGraphNode_t k6 = AppendResetNode(graph, { k1 }, AsGradient);
	cudaGraphNode_t k7 = AppendResetNode(graph, { k1 }, AdGradient);
	cudaGraphNode_t k8 = AppendResetNode(graph, { k1 }, OTGradient);
	cudaGraphNode_t k9 = AppendResetNode(graph, { k1 }, inputGradientQ);
	cudaGraphNode_t k10 = AppendResetNode(graph, { k1 }, inputGradientK);
	cudaGraphNode_t k11 = AppendResetNode(graph, { k1 }, inputGradientV);

	cudaGraphNode_t k12 = AppendMatMulPlusBatchNode(graph, { k1 }, OT, outputGradient, WOOpt.gradient, false, false, batch, false, false, true);
	cudaGraphNode_t k13 = AppendMatMulPlusBatchNode(graph, { k8 }, WO, outputGradient, OTGradient, false, true, batch, true, false, false);

	cudaGraphNode_t k14 = AppendMatMulPlusBatchNode(graph, { k7, k13 }, OTGradient, VT, AdGradient, true, false, batch * head, false, false, false);
	cudaGraphNode_t k15 = AppendMatMulPlusBatchNode(graph, { k4, k13 }, OTGradient, Ad, VTGradient, false, false, batch * head, false, false, false);

	cudaGraphNode_t k16 = dropout.AppendGraphBackward(graph, { k6, k14 });
	cudaGraphNode_t k17 = softmax.AppendGraphBackward(graph, { k5, k16 });
	cudaGraphNode_t k18;
	switch(maskType) {
		case LOOK_AHEAD : k18 = AppendLookAheadMaskBatchNode(graph, { k17 }, AGradient, batch * head, seq, 0.0f); break;
		case PADDING: k18 = AppendPaddingMaskBatchNode(graph, { k17 }, AGradient, batch * head, seq, 0.0f); break;
		case CROSS_PADDING: k18 = AppendCrossPaddingMaskBatchNode(graph, { k17 }, AGradient, batch * head, seq, 0.0f); break;
	}
	cudaGraphNode_t k19 = AppendDivInplaceNode(graph, { k18 }, AGradient, std::sqrt(dModel / head));

	cudaGraphNode_t k20 = AppendMatMulPlusBatchNode(graph, { k2, k19 }, KT, AGradient, QTGradient, false, true, batch * head, false, false, false);
	cudaGraphNode_t k21 = AppendMatMulPlusBatchNode(graph, { k3, k19 }, QT, AGradient, KTGradient, false, false, batch * head, false, false, false);

	cudaGraphNode_t k22 = AppendMatMulPlusBatchNode(graph, { k20 }, QTGradient, inputQ, WQOpt.gradient, false, false, batch, false, false, true);
	cudaGraphNode_t k23 = AppendMatMulPlusBatchNode(graph, { k21 }, KTGradient, inputK, WKOpt.gradient, false, false, batch, false, false, true);
	cudaGraphNode_t k24 = AppendMatMulPlusBatchNode(graph, { k15 }, VTGradient, inputV, WVOpt.gradient, false, false, batch, false, false, true);
	cudaGraphNode_t k25 = AppendMatMulPlusBatchNode(graph, { k9, k20 }, QTGradient, WQ, inputGradientQ, true, false, batch, false, true, false);
	cudaGraphNode_t k26 = AppendMatMulPlusBatchNode(graph, { k10, k21, k25 }, KTGradient, WK, inputGradientK, true, false, batch, false, true, false); // in case all inputGradient point to the same place
	cudaGraphNode_t k27 = AppendMatMulPlusBatchNode(graph, { k11, k15, k26 }, VTGradient, WV, inputGradientV, true, false, batch, false, true, false);

	cudaGraphNode_t k28 = SyncDependency(graph, { k12, k22, k23, k24, k25, k26, k27 });
	return k28;
}

cudaGraphNode_t MultiheadAttention::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
	cudaGraphNode_t k2 = AppendAdamOptNode(graph, { k1 }, WQ, WQOpt);
	cudaGraphNode_t k3 = AppendAdamOptNode(graph, { k1 }, WK, WKOpt);
	cudaGraphNode_t k4 = AppendAdamOptNode(graph, { k1 }, WV, WVOpt);
	cudaGraphNode_t k5 = AppendAdamOptNode(graph, { k1 }, WO, WOOpt);
	cudaGraphNode_t k6 = SyncDependency(graph, { k2, k3, k4, k5 });
	return k6;
}