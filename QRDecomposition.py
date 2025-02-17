# -*- coding: utf-8 -*-

import numpy as np


def QR_decomposition(matrix, eps = 1e-14):
    """
    The function performs QR decomposition on the matrix
    :param eps: epsilon for linear dependency, default 1e-14
    :param matrix: matrix with linear independent vectors
    :return: Q matrix, R matrix and matrix after removing linear dependent vectors
    """
    rows, cols = matrix.shape
    A = matrix.copy()
    
    Q = np.zeros((rows, cols), dtype=float) # Q in QR
    R = np.zeros((rows, cols), dtype=float)
   
    independent_columns = []
    
    for i in range(Q.shape[1]): # Iter over cols
        vector = A[:, i].copy() # get the i-th col (all the rows in i col)

        for j in range(i):
            R[j,i] = np.dot(Q[:, j], vector)
            vector -= R[j,i] * Q[:, j]
            
        norm = np.linalg.norm(vector)
        
        if norm > eps:
            R[len(independent_columns), i] = norm
            Q[:, len(independent_columns)] = vector / norm
            independent_columns.append(i) # add col num to keep track
        else:
            print(f"Column {i} is linearly dependent and removed.")  # Debugging output
        
    Q = Q[:, :len(independent_columns)]
    R = R[:len(independent_columns), :len(independent_columns)]
    A_updated = Q@R

    return A_updated, Q, R


def test_QR():
    input_matrix = np.array([
        [1, 1, 3],
        [2, 1, 4],
        [3, -1, 1],
        [2, 1, 1],
        [1, 0, 2]
    ], dtype=float)

    A_updated, Q, R = QR_decomposition(input_matrix)
    # A_updated = A_updated.astype(int)
    print("Updated Matrix A (after removing dependent columns if any):")
    print(A_updated)
    print("\nOrthonormal Matrix Q:")
    print(Q)
    print("\nUpper Triangular Matrix R:")
    print(R)
    # Verify that Q is orthonormal: Q^T * Q = I
    print("\nVerification (Q^T * Q should be identity):")
    print(np.dot(Q.T, Q))
    # Verify that QR reconstructs the (updated) A
    print("\nVerification (Q * R should reconstruct updated A):")
    print(np.dot(Q, R))


if __name__ == "__main__":
    test_QR()
