#ifndef TENSOR
#define TENSOR

#include "Header.h"

class Tensor {
public:
	Tensor();
	Tensor(const int batch, const int row, const int column) noexcept;
	Tensor(const Tensor& other) noexcept;
	Tensor(Tensor&& other) noexcept;

	Matrix& operator[](const int idx) noexcept;
	const Matrix& operator[](const int idx) const noexcept;

	Tensor& operator=(const Tensor& other) noexcept;
	Tensor& operator=(const float x) noexcept;
	Tensor operator*(const float x) const noexcept;
	Tensor operator*(const Matrix& other) const noexcept;
	Tensor operator/(const float x) const noexcept;
	Tensor operator+(const float x) const noexcept;
	Tensor operator+(const Matrix& other) const noexcept;
	Tensor operator+(const Tensor& other) const noexcept;
	Tensor operator-(const float x) const noexcept;
	Tensor operator-(const Matrix& other) const noexcept;

	friend std::ostream& operator<<(std::ostream& os, const Tensor& m);
	friend std::istream& operator>>(std::istream& is, Tensor& m);

	Tensor transpose() const noexcept;

	int batch					= 0;
	std::vector<Matrix> data	= std::vector<Matrix>();
};

#endif