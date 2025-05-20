#include "Header.h"


Tensor::Tensor() { ; }

Tensor::Tensor(const int batch, const int row, const int column) noexcept : 
	batch(batch), 
	data(std::vector<Matrix>(batch,Matrix(row,column))) { ; }

Tensor::Tensor(const Tensor& other) noexcept : batch(other.batch), data(other.data) { ; }

Tensor::Tensor(Tensor&& other) noexcept : batch(other.batch), data(std::move(other.data)) { ; }


Matrix& Tensor::operator[](const int idx) noexcept {
	return data[idx];
}

const Matrix& Tensor::operator[](const int idx) const noexcept {
	return data[idx];
}

Tensor& Tensor::operator=(const Tensor& other) noexcept {
	batch = other.batch;
	data = other.data;
	return (*this);
}

Tensor& Tensor::operator=(const float other) noexcept {
	for (int i = 0; i < data.size(); i++) {
		data[i] = other;
	}
	return (*this);
}

Tensor Tensor::operator*(const float x) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] * x;
	}
	return result;
}

Tensor Tensor::operator*(const Matrix& other) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] * other;
	}
	return result;
}

Tensor Tensor::operator/(const float x) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] / x;
	}
	return result;
}

Tensor Tensor::operator+(const float x) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] + x;
	}
	return result;
}

Tensor Tensor::operator+(const Matrix& other) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] + other;
	}
	return result;
}

Tensor Tensor::operator+(const Tensor& other) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] + other[i];
	}
	return result;
}


Tensor Tensor::operator-(const float x) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] - x;
	}
	return result;
}

Tensor Tensor::operator-(const Matrix& other) const noexcept {
	Tensor result(batch, data[0].row, data[0].column);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i] - other;
	}
	return result;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
	for (int i = 0; i < tensor.batch; i++) {
		os << tensor.data[i];
	}
	return os;
}

std::istream& operator>>(std::istream& is, Tensor& tensor) {
	for (int i = 0; i < tensor.batch; i++) {
		is >> tensor.data[i];
	}
	return is;
}

Tensor Tensor::transpose() const noexcept {
	Tensor result(batch, data[0].column, data[0].row);
	for (int i = 0; i < batch; i++) {
		result[i] = data[i].transpose();
	}
	return result;
}