#include "Header.h"

Matrix::Matrix() { ; }

Matrix::Matrix(const int row, const int column) noexcept : 
	row(row), 
	column(column), 
	data(std::vector<float>(row* column)) { 
	; }

Matrix::Matrix(const int row, const int column, const float value) noexcept : row(row), column(column), data(std::vector<float>(row* column, value)) { ; }

Matrix::Matrix(const Matrix& other) noexcept : row(other.row), column(other.column), data(other.data) { ; }

Matrix::Matrix(Matrix&& other) noexcept : row(other.row), column(other.column), data(std::move(other.data)) { ; }


float* Matrix::operator[](const int idx) noexcept {
	return &(data.data()[idx * column]);
}

const float* Matrix::operator[](const int idx) const noexcept {
	return &(data.data()[idx * column]);
}


Matrix& Matrix::operator=(const Matrix& other) noexcept {
	row = other.row;
	column = other.column;
	data = other.data;
	return (*this);
}

Matrix& Matrix::operator=(const float x) noexcept {
	for (int i = 0; i < data.size(); i++) {
		data[i] = x;
	}
	return (*this);
}

Matrix Matrix::operator*(const float x) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < data.size(); i++) {
		result.data[i] = data[i] * x;
	}
	return result;
}

Matrix Matrix::operator*(const Matrix& other) const noexcept {
	Matrix result(row, other.column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < other.column; j++) {
			for (int k = 0; k < column; k++) {
				result[i][j] += (*this)[i][k] * other[k][j];
			}
		}
	}
	return result;
}

Matrix Matrix::operator/(const float x) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < result.data.size(); i++) {
		result.data[i] = data[i] / x;
	}
	return result;
}

Matrix& Matrix::operator/=(const float x) noexcept {
	for (int i = 0; i < data.size(); i++) {
		data[i] /= x;
	}
	return (*this);
}

Matrix Matrix::operator+(const float x) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < data.size(); i++) {
		result.data[i] = data[i] + x;
	}
	return result;
}

Matrix Matrix::operator+(const Matrix& other) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < data.size(); i++) {
		result.data[i] = data[i] + other.data[i];
	}
	return result;
}

Matrix Matrix::operator-(const float x) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < data.size(); i++) {
		result.data[i] = data[i] - x;
	}
	return result;
}

Matrix Matrix::operator-(const Matrix& other) const noexcept {
	Matrix result(row, column);
	for (int i = 0; i < data.size(); i++) {
		result.data[i] = data[i] - other.data[i];
	}
	return result;
}

Matrix& Matrix::operator-=(const Matrix& other) noexcept {
	for (int i = 0; i < data.size(); i++) {
		data[i] -= other.data[i];
	}
	return (*this);
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
	for (int i = 0; i < matrix.row; i++) {
		for (int j = 0; j < matrix.column; j++) {
			os << matrix[i][j] << ' ';
		}
		os << '\n';
	}
	return os;
}

std::istream& operator>>(std::istream& is, Matrix& matrix) {
	for (int i = 0; i < matrix.row; i++) {
		for (int j = 0; j < matrix.column; j++) {
			is >> matrix[i][j];
		}
	}
	return is;
}

Matrix Matrix::lookAheadMask(const int size) noexcept {
	Matrix result(size, size);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j <= i; j++) {
			result[i][j] = 1;
		}
	}
	return result;
}

Matrix Matrix::transpose() const noexcept {
	Matrix result(column, row);
	for (int i = 0; i < column; i++) {
		for (int j = 0; j < row; j++) {
			result[i][j] = (*this)[j][i];
		}
	}
	return result;
}
