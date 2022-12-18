import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from implicit_euler_trajectory import dR, dJ, R0, J0
Sts = [[R0, J0], [R0 + 4, J0 + 4], [R0 - 4, J0 - 4]]
# ODE System


def ivpSys(s, t, dR, dJ):
    R, J = s
    T = t
    dRdt = dR(R, J, T)
    dJdt = dJ(R, J, T)
    return [dRdt, dJdt]


# Vector R', J' at t = 0 with 144 values of pair (R0, J0)
y1 = np.linspace(-6, 6, 12)
y2 = np.linspace(-6, 6, 12)
Y1, Y2 = np.meshgrid(y1, y2)
t = 0
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape
for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yvector = ivpSys([x, y], t, dR, dJ)
        u[i, j] = yvector[0]
        v[i, j] = yvector[1]
Q = plt.quiver(Y1, Y2, u, v, color='g')
# Trajectory of Sts with 200 values of t between (0,5)
tspan = np.linspace(0, 5, 200)
for elementSt, St in enumerate(Sts):
    sol = odeint(ivpSys, St, tspan, args=(dR, dJ))
    if elementSt == 0:
        color = "r"
    else:
        color = "darkgray"
    plt.plot(sol[:, 0], sol[:, 1], color, label='Trajectory')
# Figure phase portrait
x = np.linspace(-4.5, 4.5, 100)


def fx(x, a, b):
    return -a * x / b


plt.plot(x, fx(x, 2, 3), linestyle='dashed', linewidth=0.75,
         color='royalblue', label='Nullcline 1')
plt.plot(x, fx(x, 2, 3), linestyle='dashed', linewidth=0.75,
         color='magenta', label='Nullcline 2')
plt.plot(0, 0, "red", marker="o", markersize=10.0,
         color='grey', label='Fixed point')
plt.xlabel("Romeo's love for Juliet")
plt.ylabel("Juliet's love for Romeo")
plt.legend()
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.show()
