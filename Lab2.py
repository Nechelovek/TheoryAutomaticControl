import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

variant = 1;

def ode(y, t, b, c) :
    theta, omega = y
    dydt = [omega, -c * np.sin(theta)]
    return dydt

def calcODE(args, y0, dy0, ts = 10, nt = 101) :
    y0 = [y0, dy0]
    t = np.linspace(0, ts, nt)
    sol = odeint(ode, y0, t, args)
    return sol

def drawPhasePortrait(args, deltaX = 1, deltaDX = 1, startX = 0,  stopX = 5, startDX = 0, stopDX = 5, ts = 10, nt = 101):
    for y0 in range(startX, stopX, deltaX):
            for dy0 in range(startDX, stopDX, deltaDX):
                sol = calcODE(args, y0, dy0, ts, nt)
                plt.plot(sol[:, 0], sol[:, 1], 'b')
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.grid()
    plt.show()


b = 0.25
c = 5.0
args=(b, c)

#System 1
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [(-4 * x) + (0.1 * variant * x * x) - (4 * y), (1.5 * x) + y - (0.2 * variant * y * y * y)]
    return dydt
#drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 5, nt = 301)
drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#System 2
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [x + (0.5 * y) - (0.1 * variant * y * y), (0.5 * x) - (0.2 * variant * x * x) + y]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#System 3
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [(2 * x) + (0.2 * variant * x * x) + y - (0.1 * variant * y * y), x - 3 * y]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#System 4
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [-0.1 * variant * x * x + 2 * y , -3 * x -y]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#System 5
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [0.1 * x - 4 * y, 4 * x - 0.2 * variant * x * x + 0.1 * y]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#System 6
def ode(Y, t, b, c) :
    x, y = Y
    dydt = [x - 0.1 * variant * x * x - 4 * y + 0.3 * variant * y * y, 2*x + 0.2*variant*x*x - y - 0.3*variant * y * y *y]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#Task 2

#Quest 3
m = 0.2 * variant
l = 5 / variant
g = 9.81

def ode(y, t, b, c) :
    theta, omega = y
    dydt = [omega, -m * g * l * np.sin(theta)]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)

def ode(y, t, b, c) :
    theta, omega = y
    dydt = [omega, (-b * omega) - (m * g * l * np.sin(theta))]
    return dydt

drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
