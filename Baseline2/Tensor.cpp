#include "Header.h"


int getSize(const int d, const int row, const int col) {
	return d * row * ((col * 255) / 256);
}

Tensor create(const int d, const int row, const int col) {
	const int realSize = getSize(d, row, col);
	return std::make_unique<__m256[]>(realSize);
}

float* toFloat(const int size, Tensor a) {
	float* out = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size / 256; i++) {
		_mm256_storeu_ps(out + (i << 8), a.get()[i]);
	}
	return out;
}

void plus(const int size, Tensor a, Tensor b, Tensor& c) {
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_add_ps(a.get()[i], b.get()[i]);
	}
}

void plus(const int size, Tensor a, const float x, Tensor& c) {
	const Element x_m256 = _mm256_set1_ps(x);
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_add_ps(a.get()[i], x_m256);
	}
}

void sub(const int size, Tensor a, Tensor b, Tensor& c) {
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_sub_ps(a.get()[i], b.get()[i]);
	}
}

void sub(const int size, Tensor a, const float x, Tensor& c) {
	const Element x_m256 = _mm256_set1_ps(x);
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_sub_ps(a.get()[i], x_m256);
	}
}

void mul(const int size, Tensor a, Tensor b, Tensor& c) {
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_mul_ps(a.get()[i], b.get()[i]);
	}
}

void mul(const int size, Tensor a, const float x, Tensor& c) {
	const Element x_m256 = _mm256_set1_ps(x);
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_mul_ps(a.get()[i], x_m256);
	}
}

void div(const int size, Tensor a, Tensor b, Tensor& c) {
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_div_ps(a.get()[i], b.get()[i]);
	}
}

void div(const int size, Tensor a, const float x, Tensor& c) {
	const Element x_m256 = _mm256_set1_ps(x);
	for (int i = 0; i < size / 256; i++) {
		c.get()[i] = _mm256_div_ps(a.get()[i], x_m256);
	}
}

void MatMul(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c) {

}

void MatMulTransA(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c) {

}

void MatMulTransB(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c) {

}