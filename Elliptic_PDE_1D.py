import numpy as np
from NumericalTools import Gauss_Seidel

# This program is modeling numerically a physical problem with the next mathematical model:

def tridiag(l, d, r, n): # Creates tridiagonal matrix (with the exception of the coefficient A[n][n-1])
    A = [[0 for i in range(n)] for i in range(n)]
    A[0][0] = d
    A[0][1] = r
    for i in range(1, n-1):
        A[i][i] = d
        A[i][i+1] = r
        A[i][i-1] = l
    A[n-1][n-2] = 2*l
    A[n-1][n-1] = d
    return A


def main():
    # Data related to the problem
    L = 0.2
    ub = 373
    u0 = 273
    uex = 303
    alpha = 250
    bita = 5
    k = 20

    # Data related to the approximation algorithm
    n = 4
    h = L / n

    # Diagonal, left side and right side of the tridiagonal matrix
    left = -1
    right = -1
    diag = (alpha*h*h / k) + 2
    An = tridiag(left, diag, right, n)
    
    # right hand side of the linear equation
    F = [(alpha*h*h/k)*uex for i in range(n)]
    F[0] += u0
    F[n-1] +=  (2*h*bita/k) * ub
    u_0 = [((uex+ub+u0)/3) for i in range(n)]

    solution_app = Gauss_Seidel.lin_sys_resolver(An, F, u_0)
    
    return solution_app


if __name__=="__main__":
    print(main())
