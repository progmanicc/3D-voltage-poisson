#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import time
import matplotlib.pyplot as plt


# In[15]:


def gaussian_upper_triangle(a,b):
    def exchange_rows_inplace(a,i,j):
        for k in range(len(a)):
            tmp = a[i][k]
            a[i][k] = a[j][k]
            a[j][k] = tmp
        return a
    
    C = np.column_stack((a,b))

    for i in range(len(C)):
        if C[i][i] ==0:
            if i < len(C) -1:
                exchange_rows_inplace(C,i,i+1)
        else:
            diag = C[i][i]
            for k in range(i+1,len(C)):
                C[k] = C[k] - C[i]*C[k][i]/diag
    return C

def back_substitute(A,fixed=None):
    N = len(A)
    x = np.zeros(N)

    x[N-1] = A[N-1][N]/A[N-1][N-1]

    if fixed != None:
        for index in fixed:
            if index ==N-1:
                x[N-1] = fixed[index]

    for i in range(N-2,-1,-1):
        sum = 0
        for j in range(i+1,N):
            sum += A[i][j]*x[j]
        sum = A[i][N] - sum
        sum /= A[i][i]
        x[i] = sum

        if fixed != None:
            for index in fixed:
                if index == i:
                    x[i]=fixed[index]


    return x

def gaussian_elimination(A,b,fixed=None):
    C = gaussian_upper_triangle(A,b)
    return back_substitute(C,fixed)

def jacobi(A,b,epsilon = 1e-8,maxiter = 5000,omega=1, fixed = None, debug = False):
    if fixed == None:
        D = np.diag(np.diag(A))
        LU = A-D
        x = np.zeros(len(b))
        D_inv = np.diag(1/np.diag(D))
        for i in range(maxiter):
            x_new = omega*np.dot(D_inv,b-np.dot(LU,x)) + (1-omega)*x
            res = np.linalg.norm(x_new-x)
            if debug:
                print(res)
            if res< epsilon:
                return x_new
            x = x_new.copy()
        raise Exception('error')
        return x
    else:
        D = np.diag(np.diag(A))
        LU = A-D
        x = np.zeros(len(b))
        D_inv = np.diag(1/np.diag(D))
        for i in range(maxiter):
            for index in fixed:
                x[index] = fixed[index]
            x_new = omega*np.dot(D_inv,b-np.dot(LU,x)+(1-omega)*x)
            for index in fixed:
                x_new[index] = fixed[index]
            res = np.linalg.norm(x_new-x)
            if debug:
                print(f'{res}')
            if res < epsilon:
                return x_new
            x=x_new.copy()
        raise Exception("error")
        return x
               

            
            


# In[20]:


if __name__ == '__main__':
    A1 = np.array([[2,1,-1],[-3,-1,2],[-2,1,2]])
    A = np.array([[2,1,-1],[-3,-2,2],[-.5,1,2]])
    b = np.array([8,-11,-3])

    x = gaussian_elimination(A,b)


# In[22]:



# In[ ]:





# In[ ]:




