# Matrix Arithmetic Package
This is a simple 'proof of concept' for building a python library for future projects.

## Features of this library

1. Matrix/Vector Multiplication
2. Dot and Cross Product
3. Determinants
4. Inverses

## Requirements

* Python 3.6+
* pybind11
* C++11 compatible compiler

## Installation/Build

For C++ binding.

```bash
pip install pybind11
```

Build the library using ```setup.py```.

```bash
python setup.py build_ext --inplace
```

Or we can just install this

```bash
pip install .
```

## Using the package

```python
import matrix_ops  # to use this package, use this import statement.

# Create matrices
A = matrix_ops.Matrix([[1, 2], [3, 4]]) # Creating 2x2 matrix
B = matrix_ops.Matrix([[5, 6], [7, 8]])

# Matrix multiplication
C = A.multiply(B) 

# Determinant
det_A = A.determinant()

# Matrix inverse
A_inv = A.inverse()

# Vector operations
v1 = [1, 2, 3]
v2 = [4, 5, 6]

# Dot and Cross Results
dot_result = matrix_ops.dot_product(v1, v2)
cross_result = matrix_ops.cross_product(v1, v2)
```

## API Reference

### Matrix Class

```Matrix(rows, cols)``` - Create zero matrix
```Matrix(data)``` - Create matrix from 2D list
```multiply(other)``` - Matrix multiplication
```determinant()``` - Calculate determinant
```inverse()``` - Calculate matrix inverse
```get_rows()```, get_cols() - Get dimensions
```get_data()``` - Get matrix data as 2D list

### Vector Functions

```dot_product(a, b)``` - Calculate dot product
```cross_product(a, b)``` - Calculate cross product (3D vectors only)

## Files

```matrix_ops.cpp``` - Main C++ source code
```setup.py``` - Build configuration
```example.py``` - Usage examples
```README.md``` - This file