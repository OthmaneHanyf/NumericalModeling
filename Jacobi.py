import numpy as np

# This algorithm works if, and only if, ||J|| = ||D⁻¹(E+F)|| < 1
# If the matrix A has a strictly dominant diagonal then the method converges
def lin_sys_resolver(A, b, x0):
    # Get the problem dimention
    n = len(A)
    if n != len(A[0]):
        return "Error : First parameter is not a sequare matrix !"
    
    # Decompose A as D-E-F
    D = [[0 for i in range(n)] for i in range(n)]
    E = [[0 for i in range(n)] for i in range(n)]
    F = [[0 for i in range(n)] for i in range(n)]
    for i in range(len(A[0])):
        D[i][i] = A[i][i]
        if i > 0 :
            E[i][i-1] = -A[i][i-1]
        if i < len(A[0])-1: 
            F[i][i+1] = -A[i][i+1]
            
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    b = np.array(b)
    x0 = np.array(x0)
    maxIter = 100000000
    epsilon = .000000000000001

    # Initialization
    xOld = x0
    J = np.dot(np.linalg.inv(D), E+F)
    xNew = np.dot(J, xOld.T) + np.dot(np.linalg.inv(D), b.T)

    # Start the iterating process
    numIter = 0
    while (numIter < maxIter and np.dot(xNew-xOld, xNew-xOld) > epsilon):
        xOld = xNew
        xNew = np.dot(J, xOld.T) + np.dot(np.linalg.inv(D), b.T)
        numIter += 1
    return xNew
