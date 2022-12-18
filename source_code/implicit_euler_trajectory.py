import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math

def Romeo(R, new_R, new_J, new_T, dR, h):
    return R + h * dR(new_R, new_J, new_T) - new_R


def Juliet(J, new_R, new_J, new_T, dJ, h):
    return J + h * dJ(new_R, new_J, new_T) - new_J


def Time(T, new_T, h):
    return T + h - new_T


def Jacobi(R, J, T, new_R, new_J, new_T, dR, dJ, h):
    jacob = np.ones((3, 3))
    d = 1e-9

    jacob[0, 0] = (Romeo(R, (new_R + d), new_J, new_T, dR, h) - Romeo(R, new_R, new_J, new_T, dR, h)) / d
    jacob[0, 1] = (Romeo(R, new_R, (new_J + d), new_T, dR, h) - Romeo(R, new_R, new_J, new_T, dR, h)) / d
    jacob[0, 2] = (Romeo(R, new_R, new_J, (new_T + d), dR, h) - Romeo(R, new_R, new_J, new_T, dR, h)) / d

    jacob[1, 0] = (Juliet(J, (new_R + d), new_J, new_T, dJ, h) - Juliet(J, new_R, new_J, new_T, dJ, h)) / d
    jacob[1, 1] = (Juliet(J, new_R, (new_J + d), new_T, dJ, h) - Juliet(J, new_R, new_J, new_T, dJ, h)) / d
    jacob[1, 2] = (Juliet(J, new_R, new_J, (new_T + d), dJ, h) - Juliet(J, new_R, new_J, new_T, dJ, h)) / d

    jacob[2, 0] = 0
    jacob[2, 1] = 0
    jacob[2, 2] = -1

    return jacob


def Newton_Raphson(R, J, T, random_R, random_J, random_T, dR, dJ, h):
    X_init = np.ones((3, 1))
    X_init[0] = random_R
    X_init[1] = random_J
    X_init[2] = random_T

    X = np.ones((3, 1))

    error = 9e9
    tol = 1e-9

    while error > tol:
        jacob = Jacobi(R, J, T, X_init[0], X_init[1], X_init[2], dR, dJ, h)

        X[0] = Romeo(R, X_init[0], X_init[1], X_init[2], dR, h)
        X[1] = Juliet(J, X_init[0], X_init[1], X_init[2], dJ, h)
        X[2] = Time(T, X_init[2], h)

        X_new = X_init - np.matmul(inv(jacob), X)
        error = np.max(np.abs(X_new - X_init))
        X_init = X_new

    return [X_new[0], X_new[1], X_new[2]]


def Implicit_Euler(dR, dJ, R0, J0, tspan, dt):
    t = np.arange(0, tspan, dt)
    R = np.zeros(len(t))
    J = np.zeros(len(t))
    T = np.zeros(len(t))

    R[0] = R0
    J[0] = J0
    T[0] = 0

    random_R = 10
    random_J = 10
    random_T = dt

    for i in range(1, len(t)):
        R[i], J[i], T[i] = Newton_Raphson(R[i - 1], J[i - 1], T[i - 1], random_R, random_J, random_T, dR, dJ, dt)

        random_R = R[i]
        random_J = J[i]
        random_T = T[i]

    return [t, R, J]


def dR(R, J, T):
    return -4 * R + R * J


def dJ(R, J, T):
    return R * R - 3 * J


R0 = 2
J0 = 3

t, R, J = Implicit_Euler(dR, dJ, R0, J0, 10, 0.001)
plt.plot(t, R)
plt.plot(t, J)
plt.xlabel("Time")
plt.ylabel("Love for the other")
plt.legend(["Romeo's", "Juliet's"])
plt.show()