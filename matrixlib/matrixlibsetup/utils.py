def is_square(matrix):
    """Check if a matrix is square."""
    return len(matrix) > 0 and len(matrix) == len(matrix[0])

def is_vector(matrix):
    """Check if input is a vector (1D matrix)."""
    return len(matrix) > 0 and isinstance(matrix[0], (int, float))