#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cmath>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Matrix multiplication (matrix-matrix or matrix-vector)
Matrix matrix_multiply(const Matrix& a, const Matrix& b) {
    if (a.empty() || b.empty() || a[0].size() != b.size())
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    Matrix result(a.size(), Vector(b[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b[0].size(); ++j)
            for (size_t k = 0; k < b.size(); ++k)
                result[i][j] += a[i][k] * b[k][j];
    return result;
}

// Matrix inverse using Gaussian elimination
Matrix inverse(const Matrix& a) {
    if (a.empty() || a.size() != a[0].size())
        throw std::invalid_argument("Matrix must be square and non-empty");
    size_t n = a.size();
    Matrix result(n, Vector(n, 0.0));
    Matrix temp = a;
    for (size_t i = 0; i < n; ++i) result[i][i] = 1.0;
    for (size_t i = 0; i < n; ++i) {
        double pivot = temp[i][i];
        if (std::abs(pivot) < 1e-10) throw std::runtime_error("Matrix is singular");
        for (size_t j = 0; j < n; ++j) {
            temp[i][j] /= pivot;
            result[i][j] /= pivot;
        }
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = temp[k][i];
                for (size_t j = 0; j < n; ++j) {
                    temp[k][j] -= factor * temp[i][j];
                    result[k][j] -= factor * result[i][j];
                }
            }
        }
    }
    return result;
}

// Determinant using LU decomposition
double determinant(const Matrix& a) {
    if (a.empty() || a.size() != a[0].size())
        throw std::invalid_argument("Matrix must be square and non-empty");
    size_t n = a.size();
    Matrix temp = a;
    double det = 1.0;
    for (size_t i = 0; i < n; ++i) {
        double pivot = temp[i][i];
        if (std::abs(pivot) < 1e-10) return 0.0;
        for (size_t k = i + 1; k < n; ++k) {
            double factor = temp[k][i] / pivot;
            for (size_t j = i; j < n; ++j)
                temp[k][j] -= factor * temp[i][j];
        }
        det *= pivot;
    }
    return det;
}

// Dot product of two vectors
double dot_product(const Vector& a, const Vector& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vectors must have the same length");
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

// Cross product for 3D vectors
Vector cross_product(const Vector& a, const Vector& b) {
    if (a.size() != 3 || b.size() != 3)
        throw std::invalid_argument("Cross product requires 3D vectors");
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Matrix or vector addition
Matrix add(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size() || (a.empty() || b.empty()) || a[0].size() != b[0].size())
        throw std::invalid_argument("Invalid dimensions for addition");
    Matrix result = a;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            result[i][j] += b[i][j];
    return result;
}

// Matrix or vector subtraction
Matrix subtract(const Matrix& a, const Matrix& b) {
    if (a.size() != b.size() || (a.empty() || b.empty()) || a[0].size() != b[0].size())
        throw std::invalid_argument("Invalid dimensions for subtraction");
    Matrix result = a;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            result[i][j] -= b[i][j];
    return result;
}

// Solve linear system Ax = b using Gaussian elimination
Vector solve(const Matrix& a, const Vector& b) {
    if (a.empty() || a.size() != a[0].size() || a.size() != b.size())
        throw std::invalid_argument("Invalid dimensions for equation solving");
    size_t n = a.size();
    Matrix temp = a;
    Vector result = b;
    for (size_t i = 0; i < n; ++i) {
        double pivot = temp[i][i];
        if (std::abs(pivot) < 1e-10) throw std::runtime_error("Matrix is singular");
        for (size_t j = i; j < n; ++j) temp[i][j] /= pivot;
        result[i] /= pivot;
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = temp[k][i];
                for (size_t j = i; j < n; ++j)
                    temp[k][j] -= factor * temp[i][j];
                result[k] -= factor * result[i];
            }
        }
    }
    return result;
}

// Scalar multiplication
Matrix scalar_multiply(const Matrix& a, double scalar) {
    Matrix result = a;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            result[i][j] *= scalar;
    return result;
}

// Matrix transpose
Matrix transpose(const Matrix& a) {
    if (a.empty()) throw std::invalid_argument("Matrix cannot be empty");
    Matrix result(a[0].size(), Vector(a.size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            result[j][i] = a[i][j];
    return result;
}

PYBIND11_MODULE(matrix, m) {
    m.doc() = "C++ module for matrix and vector operations";
    m.def("matrix_multiply", &matrix_multiply, "Multiply two matrices or a matrix and vector");
    m.def("inverse", &inverse, "Compute the inverse of a square matrix");
    m.def("determinant", &determinant, "Compute the determinant of a square matrix");
    m.def("dot_product", &dot_product, "Compute the dot product of two vectors");
    m.def("cross_product", &cross_product, "Compute the cross product of two 3D vectors");
    m.def("add", &add, "Add two matrices or vectors");
    m.def("subtract", &subtract, "Subtract two matrices or vectors");
    m.def("solve", &solve, "Solve a linear system Ax = b");
    m.def("scalar_multiply", &scalar_multiply, "Multiply a matrix or vector by a scalar");
    m.def("transpose", &transpose, "Transpose a matrix");
}