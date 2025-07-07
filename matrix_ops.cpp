#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    Matrix(const std::vector<std::vector<double>>& input) {
        rows = input.size();
        cols = input[0].size();
        data = input;
    }
    
    // Access elements
    double& operator()(size_t i, size_t j) {
        return data[i][j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        return data[i][j];
    }
    
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    // Get raw data for Python
    std::vector<std::vector<double>> getData() const {
        return data;
    }
    
    // Matrix multiplication
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Determinant (using cofactor expansion)
    double determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for determinant");
        }
        
        if (rows == 1) return data[0][0];
        if (rows == 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        
        double det = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            Matrix minor = getMinor(0, j);
            det += (j % 2 == 0 ? 1 : -1) * data[0][j] * minor.determinant();
        }
        return det;
    }
    
    // Matrix inverse (using Gauss-Jordan elimination)
    Matrix inverse() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for inverse");
        }
        
        double det = determinant();
        if (std::abs(det) < 1e-10) {
            throw std::invalid_argument("Matrix is singular (determinant = 0)");
        }
        
        // Create augmented matrix [A|I]
        Matrix augmented(rows, 2 * cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                augmented(i, j) = data[i][j];
                augmented(i, j + cols) = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Gauss-Jordan elimination
        for (size_t i = 0; i < rows; ++i) {
            // Find pivot
            double pivot = augmented(i, i);
            if (std::abs(pivot) < 1e-10) {
                throw std::invalid_argument("Matrix is singular");
            }
            
            // Scale row
            for (size_t j = 0; j < 2 * cols; ++j) {
                augmented(i, j) /= pivot;
            }
            
            // Eliminate column
            for (size_t k = 0; k < rows; ++k) {
                if (k != i) {
                    double factor = augmented(k, i);
                    for (size_t j = 0; j < 2 * cols; ++j) {
                        augmented(k, j) -= factor * augmented(i, j);
                    }
                }
            }
        }
        
        // Extract inverse matrix
        Matrix inv(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                inv(i, j) = augmented(i, j + cols);
            }
        }
        return inv;
    }
    
private:
    Matrix getMinor(size_t row, size_t col) const {
        Matrix minor(rows - 1, cols - 1);
        size_t mi = 0, mj = 0;
        for (size_t i = 0; i < rows; ++i) {
            if (i == row) continue;
            mj = 0;
            for (size_t j = 0; j < cols; ++j) {
                if (j == col) continue;
                minor(mi, mj) = data[i][j];
                mj++;
            }
            mi++;
        }
        return minor;
    }
};

// Vector operations
std::vector<double> dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same size for dot product");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return {result};
}

std::vector<double> crossProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product only defined for 3D vectors");
    }
    
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Python bindings
PYBIND11_MODULE(matrix_ops, m) {
    m.doc() = "Simple matrix operations library";
    
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<const std::vector<std::vector<double>>&>())
        .def("__call__", [](Matrix& m, size_t i, size_t j) -> double& {
            return m(i, j);
        }, py::return_value_policy::reference)
        .def("get_rows", &Matrix::getRows)
        .def("get_cols", &Matrix::getCols)
        .def("get_data", &Matrix::getData)
        .def("multiply", &Matrix::multiply)
        .def("determinant", &Matrix::determinant)
        .def("inverse", &Matrix::inverse)
        .def("__repr__", [](const Matrix& m) {
            std::string result = "Matrix([\n";
            auto data = m.getData();
            for (const auto& row : data) {
                result += "  [";
                for (size_t i = 0; i < row.size(); ++i) {
                    result += std::to_string(row[i]);
                    if (i < row.size() - 1) result += ", ";
                }
                result += "]\n";
            }
            result += "])";
            return result;
        });
    
    m.def("dot_product", &dotProduct, "Calculate dot product of two vectors");
    m.def("cross_product", &crossProduct, "Calculate cross product of two 3D vectors");
}