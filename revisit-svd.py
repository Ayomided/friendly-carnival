import numpy as np

M = np.array([
    [1, 1, 1, 0, 0],
    [3, 3, 3, 0, 0],
    [4, 4, 4, 0, 0],
    [5, 5, 5, 0, 0],
    [0, 2, 0, 4, 4],
    [0, 0, 0, 5, 5],
    [0, 1, 0, 2, 2]
])


# Perform SVD
U, s, Vt = np.linalg.svd(M, full_matrices=True)

# Create Sigma matrix
Sigma = np.zeros(M.shape)
np.fill_diagonal(Sigma, s)


# for j in U:
#     for i in j:
#         if i == 0.0 | -0.0:


print("Original Matrix M:")
print(M)
print("\nLeft Singular Vectors (U):")
print(U)
print("\nSingular Values (s):")
print(s)
print("\nSigma Matrix:")
print(Sigma)
print("\nRight Singular Vectors Transposed (V^T):")
print(Vt)