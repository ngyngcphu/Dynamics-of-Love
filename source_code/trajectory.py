import numpy as np
import matplotlib.pyplot as plt
import math

pi = math.pi
a = -3
b = -13
c = 5
d = -1
R0 = 2
J0 = -2


def delta(a, b, c, d):
    return pow((a + d), 2) - 4 * (a * d - b * c)


def lamda(a, b, c, d):
    deltaValue = delta(a, b, c, d)
    if deltaValue > 0:
        deltaSqrt = math.sqrt(deltaValue)
        lamda1 = (a + d - deltaSqrt) / 2
        lamda2 = (a + d + deltaSqrt) / 2
        return [lamda1, lamda2]
    elif deltaValue == 0:
        x = (a + d) / 2
        return x
    else:
        deltaValue = abs(deltaValue)
        deltaSqrt = math.sqrt(deltaValue)
        a1 = (a + d) / 2
        b1 = deltaSqrt / 2
        return [a1, b1]


def F1(a, b, R0, J0, k1, k2):
    D = k2 - k1
    D1 = R0 * (k2 - a) - J0 * b
    D2 = -R0 * (k1 - a) + J0 * b
    return [D1 / D, D2 / D]


def F2(a, b, R0, J0, k):
    x = J0 * b - (k - a) * R0
    return x


def F3(a, b, R0, J0, m, n):
    x = ((b * J0) - ((m - a) * R0)) / n
    return x


def fR(t, a, b, c, d, R0, J0):
    deltaValue = delta(a, b, c, d)
    if deltaValue > 0:
        k1 = lamda(a, b, c, d)[0]
        k2 = lamda(a, b, c, d)[1]
        c1 = F1(a, b, R0, J0, k1, k2)[0]
        c2 = F1(a, b, R0, J0, k1, k2)[1]
        R = c1 * math.exp(k1 * t) + c2 * math.exp(k2 * t)
        return R
    elif deltaValue == 0:
        k = lamda(a, b, c, d)
        c2 = F2(a, b, R0, J0, k)
        R = R0 * math.exp(k * t) + c2 * math.exp(k * t) * t
        return R
    else:
        m = lamda(a, b, c, d)[0]
        n = lamda(a, b, c, d)[1]
        x = F3(a, b, R0, J0, m, n)
        R = math.exp(m * t) * (R0 * math.cos((n * t)) + x * math.sin((n * t)))
        return R


def fJ(t, a, b, c, d, R0, J0):
    deltaValue = delta(a, b, c, d)
    if deltaValue > 0:
        k1 = lamda(a, b, c, d)[0]
        k2 = lamda(a, b, c, d)[1]
        c1 = F1(a, b, R0, J0, k1, k2)[0]
        c2 = F1(a, b, R0, J0, k1, k2)[1]
        R = fR(t, a, b, c, d, R0, J0)
        dRdt = c1 * k1 * math.exp(k1 * t) + c2 * k2 * math.exp(k2 * t)
        J = (1 / b) * (dRdt - a * R)
        return J
    elif deltaValue == 0:
        k = lamda(a, b, c, d)
        c2 = F2(a, b, R0, J0, k)
        J = (1 / b) * math.exp(k * t) * (J0 * b + (k - a) * c2 * t)
        return J
    else:
        m = lamda(a, b, c, d)[0]
        n = lamda(a, b, c, d)[1]
        c2 = F3(a, b, R0, J0, m, n)
        R = fR(t, a, b, c, d, R0, J0)
        dRdt = m * math.exp(m * t) * (R0 * math.cos((n * t)) + c2 * math.sin((n * t))) + \
            math.exp(m * t) * (- (R0 * n * math.sin((n * t))) +
                               (c2 * n * math.cos((n * t))))
        J = (1 / b) * (dRdt - a * R)
        return J


t = np.linspace(0, 10, 100)
Rt = np.vectorize(fR)
Jt = np.vectorize(fJ)
plt.plot(t, Rt(t, a, b, c, d, R0, J0))
plt.plot(t, Jt(t, a, b, c, d, R0, J0))
plt.xlabel("Time")
plt.ylabel("Love for the other")
plt.legend(["Romeo's", "Juliet's"])
plt.show()
