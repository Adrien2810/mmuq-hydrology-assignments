import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# 0. Dummy model definition
#    f(x1, x2, x3) = x1 + 2 x2^2 + sin(pi x3)
#    Inputs X are assumed in [0, 1]
# ============================================================

def dummy_model(X):
    """
    X: array of shape (N, k)
    returns: array y of shape (N,)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    y = x1 + 2.0 * x2**2 + np.sin(np.pi * x3)
    return y


# ============================================================
# 1. Define problem size and sampling
# ============================================================

# Number of parameters
k = 3  # e.g. 3 parameters

# Sample size (number of radial starting points)
N = 1000  # choose reasonably large in practice

# For simplicity: use plain random sampling in [0,1]
# In a serious application, you would use low-discrepancy Sobol sequences.
rng = np.random.default_rng(seed=123)  # reproducible

# Matrix A and B: each is N x k
A = rng.random(size=(N, k))
B = rng.random(size=(N, k))


# ============================================================
# 2. Build the hybrid matrices A_Bi   (A^(i)_B)
#    For each parameter i:
#    - copy A
#    - replace column i of A with column i of B
# ============================================================

def build_A_Bi(A, B, i):
    """
    Construct A_Bi: same as A, but with column i from B.
    
    Parameters
    ----------
    A, B : arrays, shape (N, k)
    i    : index of parameter to replace (0-based)
    
    Returns
    -------
    A_Bi : array, shape (N, k)
    """
    A_Bi = A.copy()
    A_Bi[:, i] = B[:, i]
    return A_Bi

# Build list of all A_Bi matrices
A_Bi_list = [build_A_Bi(A, B, i) for i in range(k)]


# ============================================================
# 3. Run the model on all needed matrices
#    We need:
#    - Y_A = f(A)
#    - Y_B = f(B)
#    - Y_A_Bi[i] = f(A_Bi_list[i])
# ============================================================

Y_A = dummy_model(A)          # shape (N,)
Y_B = dummy_model(B)          # shape (N,)
Y_A_Bi = [dummy_model(M) for M in A_Bi_list]  # list of k arrays, each shape (N,)
#print(Y_A_Bi)

# Estimate output variance Var(Y)

V_Y = np.var(Y_A, ddof=1)  # unbiased estimator


# ============================================================
# 4. Compute Sobol indices using Jansen/Saltelli estimators
#    Using configuration: A, B, A_Bi
#
#    Total-effect index (Jansen 1999, Saltelli 2010):
#    S_Ti ≈ (1 / (2N)) * sum_j [ (Y_A[j] - Y_A_Bi[i][j])^2 ] / Var(Y)
#
#    First-order index (Saltelli 2002/2010):
#    S_i ≈ (1 / N) * sum_j [ Y_B[j] * (Y_A_Bi[i][j] - Y_A[j]) ] / Var(Y)
# ============================================================

S_first = np.zeros(k)
S_total = np.zeros(k)

for i in range(k):
    Y_A_Bi_i = Y_A_Bi[i]

    # ----- Total-effect index S_Ti (Jansen estimator) -----
    # numerator_T = (1 / (2N)) * sum (Y_A - Y_A_Bi)^2
    diff = Y_A - Y_A_Bi_i
    numerator_T = np.mean(diff**2) / 2.0
    S_total[i] = numerator_T / V_Y

    # ----- First-order index S_i (Saltelli estimator) -----
    # numerator_S = (1 / N) * sum [ Y_B * (Y_A_Bi - Y_A) ]
    numerator_S = np.mean(Y_B * (Y_A_Bi_i - Y_A))
    S_first[i] = numerator_S / V_Y

# ============================================================
# 5. Print results
# ============================================================

param_names = ["X1", "X2", "X3"]  

print("Estimated Sobol indices for dummy_model:")
print(f"Output variance Var(Y) ≈ {V_Y:.4f}\n")



# Example:
# param_names = ["X1", "X2", "X3"]
# S_first = [...]
# S_total = [...]

x = np.arange(len(param_names))
width = 0.20  # thickness of bars

plt.figure(figsize=(8,5))
plt.bar(x - width/2, S_first, width, label="First-order $S_i$")
plt.bar(x + width/2, S_total, width, label="Total-effect $S_{Ti}$")

plt.xticks(x, param_names, fontsize=12)
plt.ylabel("Sensitivity index value", fontsize=12)
plt.title("Sobol Sensitivity Indices (Dummy Model)", fontsize=14)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()




