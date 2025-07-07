"""
Test suite for matrix_ops library
Run with: python test_matrix_ops.py
"""

import sys
import math

try:
    import matrix_ops
    print(" Successfully imported matrix_ops")
except ImportError as e:
    print(f" Failed to import matrix_ops: {e}")
    print("Make sure to build the library first:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)

def test_matrix_creation():
    """Test matrix creation and basic operations"""
    print("\n=== Testing Matrix Creation ===")
    
    # Test creating from list
    A = matrix_ops.Matrix([[1, 2], [3, 4]])
    assert A.get_rows() == 2
    assert A.get_cols() == 2
    print(" Matrix creation from list")
    
    # Test creating zero matrix
    B = matrix_ops.Matrix(3, 3)
    assert B.get_rows() == 3
    assert B.get_cols() == 3
    print(" Zero matrix creation")
    
    # Test data access
    data = A.get_data()
    assert data[0][0] == 1
    assert data[0][1] == 2
    assert data[1][0] == 3
    assert data[1][1] == 4
    print(" Matrix data access")

def test_matrix_multiplication():
    """Test matrix multiplication"""
    print("\n=== Testing Matrix Multiplication ===")
    
    A = matrix_ops.Matrix([[1, 2], [3, 4]])
    B = matrix_ops.Matrix([[5, 6], [7, 8]])
    
    C = A.multiply(B)
    data = C.get_data()
    
    # Expected result: [[19, 22], [43, 50]]
    assert data[0][0] == 19
    assert data[0][1] == 22
    assert data[1][0] == 43
    assert data[1][1] == 50
    print(" 2x2 matrix multiplication")
    
    # Test dimension mismatch
    try:
        D = matrix_ops.Matrix([[1, 2, 3]])  # 1x3
        E = matrix_ops.Matrix([[1], [2]])   # 2x1
        D.multiply(E)  # Should fail
        assert False, "Should have raised dimension error"
    except Exception as e:
        print(" Dimension mismatch handling")

def test_determinant():
    """Test determinant calculation"""
    print("\n=== Testing Determinant ===")
    
    # 2x2 matrix
    A = matrix_ops.Matrix([[1, 2], [3, 4]])
    det_A = A.determinant()
    assert abs(det_A - (-2)) < 1e-10
    print(" 2x2 determinant")
    
    # 3x3 matrix
    B = matrix_ops.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    det_B = B.determinant()
    assert abs(det_B) < 1e-10  # This matrix has determinant 0
    print(" 3x3 determinant (singular)")
    
    # 3x3 non-singular matrix
    C = matrix_ops.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    det_C = C.determinant()
    assert abs(det_C - 4) < 1e-10
    print(" 3x3 determinant (non-singular)")
    
    # 1x1 matrix
    D = matrix_ops.Matrix([[5]])
    det_D = D.determinant()
    assert abs(det_D - 5) < 1e-10
    print(" 1x1 determinant")

def test_matrix_inverse():
    """Test matrix inverse calculation"""
    print("\n=== Testing Matrix Inverse ===")
    
    # 2x2 invertible matrix
    A = matrix_ops.Matrix([[1, 2], [3, 4]])
    A_inv = A.inverse()
    
    # Verify by multiplying A * A_inv should give identity
    identity = A.multiply(A_inv)
    data = identity.get_data()
    
    assert abs(data[0][0] - 1) < 1e-10
    assert abs(data[0][1]) < 1e-10
    assert abs(data[1][0]) < 1e-10
    assert abs(data[1][1] - 1) < 1e-10
    print(" 2x2 matrix inverse")
    
    # Test singular matrix
    try:
        B = matrix_ops.Matrix([[1, 2], [2, 4]])  # Singular
        B.inverse()
        assert False, "Should have raised singular matrix error"
    except Exception as e:
        print(" Singular matrix handling")
    
    # 3x3 invertible matrix
    C = matrix_ops.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    C_inv = C.inverse()
    identity3 = C.multiply(C_inv)
    data3 = identity3.get_data()
    
    # Check if result is close to identity matrix
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert abs(data3[i][j] - expected) < 1e-10
    print(" 3x3 matrix inverse")

def test_dot_product():
    """Test vector dot product"""
    print("\n=== Testing Dot Product ===")
    
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    result = matrix_ops.dot_product(v1, v2)
    assert result[0] == 32  # 1*4 + 2*5 + 3*6 = 32
    print(" 3D vector dot product")
    
    # Test 2D vectors
    v3 = [1, 2]
    v4 = [3, 4]
    result2 = matrix_ops.dot_product(v3, v4)
    assert result2[0] == 11  # 1*3 + 2*4 = 11
    print(" 2D vector dot product")
    
    # Test dimension mismatch
    try:
        v5 = [1, 2]
        v6 = [3, 4, 5]
        matrix_ops.dot_product(v5, v6)
        assert False, "Should have raised dimension error"
    except Exception as e:
        print(" Dot product dimension mismatch handling")

def test_cross_product():
    """Test vector cross product"""
    print("\n=== Testing Cross Product ===")
    
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    
    result = matrix_ops.cross_product(v1, v2)
    assert result[0] == 0
    assert result[1] == 0
    assert result[2] == 1
    print(" Basic cross product")
    
    # Test general case
    v3 = [1, 2, 3]
    v4 = [4, 5, 6]
    result2 = matrix_ops.cross_product(v3, v4)
    # Expected: [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    assert result2[0] == -3
    assert result2[1] == 6
    assert result2[2] == -3
    print(" General cross product")
    
    # Test wrong dimensions
    try:
        v5 = [1, 2]
        v6 = [3, 4]
        matrix_ops.cross_product(v5, v6)
        assert False, "Should have raised dimension error"
    except Exception as e:
        print(" Cross product dimension handling")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Testing Edge Cases ===")
    
    # Test non-square matrix determinant
    try:
        A = matrix_ops.Matrix([[1, 2, 3], [4, 5, 6]])
        A.determinant()
        assert False, "Should have raised non-square error"
    except Exception as e:
        print(" Non-square determinant error")
    
    # Test non-square matrix inverse
    try:
        A = matrix_ops.Matrix([[1, 2, 3], [4, 5, 6]])
        A.inverse()
        assert False, "Should have raised non-square error"
    except Exception as e:
        print(" Non-square inverse error")
    
    # Test empty vectors
    try:
        matrix_ops.dot_product([], [])
        print(" Empty vector handling")
    except Exception as e:
        print(" Empty vector error handling")

def run_all_tests():
    """Run all tests"""
    print("Running matrix_ops library tests...")
    
    try:
        test_matrix_creation()
        test_matrix_multiplication()
        test_determinant()
        test_matrix_inverse()
        test_dot_product()
        test_cross_product()
        test_edge_cases()
        
        print("\n" + "="*50)
        print("====ALL TESTS PASSED====")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()