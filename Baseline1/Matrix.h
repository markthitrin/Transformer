#ifndef MATRIX
#define MATRIX

#include "Header.h"

class Matrix {
public:
	Matrix();
	Matrix(const int row, const int column) noexcept;
	Matrix(const int row, const int column, const float value) noexcept;
	Matrix(const Matrix& other) noexcept;
	Matrix(Matrix&& other) noexcept;

	float* operator[](const int idx) noexcept;
	const float* operator[](const int idx) const noexcept;
	
	Matrix& operator=(const Matrix& other) noexcept;
	Matrix& operator=(const float x) noexcept;
	Matrix operator*(const float x) const noexcept;
	Matrix operator*(const Matrix& other) const noexcept;
	Matrix operator/(const float x) const noexcept;
	Matrix& operator/=(const float x) noexcept;
	Matrix operator+(const float x) const noexcept;
	Matrix operator+(const Matrix& other) const noexcept;
	Matrix operator-(const float x) const noexcept;
	Matrix operator-(const Matrix& other) const noexcept;
	Matrix& operator-=(const Matrix& other) noexcept;

	friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
	friend std::istream& operator>>(std::istream& is, Matrix& m);

	static Matrix lookAheadMask(const int size) noexcept;

	Matrix transpose() const noexcept;

	std::vector<float> data	= std::vector<float>(0);
	int row					= 0;
	int column				= 0;
};

#endif // ! MATRIX