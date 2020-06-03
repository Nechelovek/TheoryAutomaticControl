import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

variant = 1;

def ode(y, t):
    theta, omega = y
    dydt = [omega, - 5 * np.sin(theta)]
    return dydt

def calcODE(y0, dy0, ts = 10, nt = 101):
    y0 = [y0, dy0]
    t = np.linspace(0, ts, nt)
    sol = odeint(ode, y0, t)
    return sol

def drawPhasePortrait(left = -10, right = 10, bottom = -10, ytop = 10, deltaX = 1, deltaDX = 1, startX = 0,  stopX = 5, startDX = 0, stopDX = 5, ts = 10, nt = 101):
    for y0 in range(startX, stopX, deltaX):
            for dy0 in range(startDX, stopDX, deltaDX):
                sol = calcODE(y0, dy0, ts, nt)
                plt.plot(sol[:, 0], sol[:, 1], 'b')
    plt.xlabel('dx')
    plt.ylabel('x')
    plt.xlim(left, right)
    plt.ylim(bottom, ytop)
    plt.grid()
    plt.show()

#System 1
# non lineary system

def ode(Y, t) :
    x, y = Y
    dydt = [(-4 * x) + (0.1 * variant * (x ** 2)) - (4 * y), (1.5 * x) + y - (0.2 * variant * y ** 3)]
    return dydt

drawPhasePortrait(-5, 55, -10, 15, 4 , 4, -10, 55, -10, 15, ts = 1, nt = 500)
drawPhasePortrait(-2, 2, -2, 2, 1 , 1, -3, 4, -3, 3, ts = 1.5, nt = 201)
#drawPhasePortrait(-4, 4, -5, 5, 1, 1, -8, 8, -4, -2, ts = 5, nt = 200)

# lineary system
def ode(Y, t):
    x, y = Y
    dydt = [-4 * x - 4 * y, 1.5 * x + y]
    return dydt

drawPhasePortrait(-2, 2, -2, 2, 1, 1, -3, 4, -3, 3, ts=2, nt=201)
drawPhasePortrait(args, 1, 1, -1, 2, -5,5, ts = 0.5, nt = 150)


# #System 2
# #non lineary system
def ode(Y, t):
    x, y = Y
    dydt = [2 * x + 0.6 * (x ** 2) + y - 0.3 * (y ** 2), x - 3 * y]
    return dydt

drawPhasePortrait(-7, 7, -10, 10, 1, 1, -10, 10, -7, 7, ts = 1, nt = 500)

# lineary system
def ode(Y, t):
    x, y = Y
    dydt = [x + 0.5 * y, 0.5 * x + y]
    return dydt

drawPhasePortrait(-7, 7, -10, 10, 1, 1, -10, 10, -5, 5, ts = 1, nt = 301)
# drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)

#
# #System 3
# def ode(y, t) :
#     x, y = y
#     dydt = [(2 * x) + (0.2 * variant * x * x) + y - (0.1 * variant * y * y), x - 3 * y]
#     return dydt
#
# drawPhasePortrait(-7, 7, -4, 4,  1, 1, -10, 5, -10, 10, ts = 0.5, nt = 301)
#
# def ode(y, t) :
#     x, y = y
#     dydt = [(2 * x) + y, x - 3 * y]
#     return dydt
#
# drawPhasePortrait(-7.5, 7.5, -7, 6,  1, 1, -5, 5, -7, 7, ts = 0.5, nt = 301)
# #drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#
# #System 4
# def ode(y, t) :
#     x, y = y
#     dydt = [-0.1 * variant * x * x + 2 * y , -3 * x -y]
#     return dydt
#
# drawPhasePortrait(-5 , 4, -5, 5, 1, 1, -5, 5, -5, 5, ts = 0.6, nt = 201)
#
# def ode(y, t) :
#     x, y = y
#     dydt = [2 * y , -3 * x - y]
#     return dydt
#
# drawPhasePortrait(-5 , 5, -5, 5, 1, 1, -5, 5, -5, 5, ts = 1, nt = 201)
#
#
# #System 5
# def ode(y, t) :
#     x, y = y
#     dydt = [0.1 * x - 4 * y, 4 * x - 0.2 * variant * x * x + 0.1 * y]
#     return dydt
#
# drawPhasePortrait(-12, 12, -10, 10, 1, 1, -10, 10, -10, 10, ts = 0.2, nt = 201)
#
# def ode(y, t) :
#     x, y = y
#     dydt = [0.1 * x - 4 * y, 4 * x + 0.1 * y]
#     return dydt
#
# drawPhasePortrait(-12, 12, -10, 10, 1, 1, -10, 10, -10, 10, ts = 0.2, nt = 201)
# #drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#
#
# #System 6
# #non lineary system
# def ode(Y, t) :
#     x, y = Y
#     dydt = [x - 0.1 * variant * x * x - 4 * y + 0.3 * variant * y * y, 2*x + 0.2*variant*x*x - y - 0.3*variant * y * y * y]
#     return dydt
#
# drawPhasePortrait(-10, 6, -4.5, 4.5, 1, 1, -8, 5, -5, 5, ts = 0.35, nt = 301)
# #drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#
# def ode(Y, t) :
#     x, y = Y
#     dydt = [x - 4 * y, 2*x - y]
#     return dydt
#
# drawPhasePortrait(-12, 8, -6, 4.5, 1, 1, -8, 5, -5, 5, ts = 0.35, nt = 301)
# #drawPhasePortrait(args, 1, 1, -5, 5, -4, 4, ts = 0.5, nt = 301)
#
#
# # #Task 2
# def dead_zone_scalar(x, wight = 0.4):
#     if np.abs(x) < wight:
#         return 0
#     elif x > 0:
#         return x - wight
#     else:
#         return x + wight
#
# dead_zone = np.vectorize(dead_zone_scalar, otypes = [np.float], excluded = ['width'])
#
#
# def ode(Y, t):
#     x, y = Y
#     dydt = [y, np.sign(dead_zone(1 - x)) - y]
#     return dydt
#
# drawPhasePortrait(-1, 3, -6, 4.5, 1, 1, -3, 4, -5, 5, ts = 4, nt = 301)
#
#
# #Quest 3
# m = 0.2 * variant
# l = 5 / variant
# g = 9.81
# b = 0.1 + 0.015 * variant
#
# def ode(y, t) :
#     theta, omega = y
#     dydt = [omega, -0.2 * 9.81 * 5 * np.sin(theta)]
#     return dydt
#
# drawPhasePortrait(-10, 10, -8, 8, 1, 1, -5, 5, -4, 4, ts = 5, nt = 301)
#
# def ode(y, t) :
#     theta, omega = y
#     dydt = [omega, (-0.115 * omega) - (0.2 * 9.81 * 5 * np.sin(theta))]
#     return dydt
#
# drawPhasePortrait(-10, 10, -8, 8, 1, 1, -8, 8, -4, 4, ts = 5, nt = 1000)
#
# #Quest 4
# m1 = variant
# m2 = variant / 2
# m3 = 2 * variant
#
# def odeVanDerPaul(Y, t, mu):
#     x, y = Y
#     dydt = [y, mu * (1 - (x ** 2)) * y - x]
#     return dydt
#
# def calcODEVanDerPaul(args, y0, dy0, ts=10, nt=101):
#     y0 = [y0, dy0]
#     t = np.linspace(0, ts, nt)
#     sol = odeint(odeVanDerPaul, y0, t, args)
#     return sol
#
# def vanDerPaul(args, deltaX=1, deltaDX=1, startX=0, stopX=5, startDX=0, stopDX=5, ts=10, nt=101):
#     for y0 in range(startX, stopX, deltaX):
#         for dy0 in range(startDX, stopDX, deltaDX):
#             sol = calcODEVanDerPaul(args, y0, dy0, ts, nt)
#             plt.plot(sol[:, 0], sol[:, 1], 'b')
#     plt.xlabel('x')
#     plt.ylabel('dx/dt')
#     plt.xlim()
#     plt.ylim()
#     plt.grid()
#     plt.show()
#
# args = (m1, )
# vanDerPaul(args, 1, 1, -4, 5, -5, 5, ts = 20, nt = 2001)
#
# args = (m1, )
# vanDerPaul(args, 1, 1, -4, 5, -5, 5, ts = 20, nt = 2001)
#
# args = (m3, )
# vanDerPaul(args, 1, 1, -4, 5, -5, 5, ts = 20, nt = 2001)

# # #Quest 5
# def ode(y, t, sigma, r, b):
#     x, y, z = y
#     dxdt = sigma * (y - x)
#     dydt = x * (r - z) - y
#     dzdt = x * y - b * z
#     return [dxdt, dydt, dzdt]
#
#
# def calcODE(args, x, y, z, ts=10, nt=101):
#     y0 = [x, y, z]
#     t = np.linspace(0, ts, nt)
#     sol = odeint(ode, y0, t, args)
#     return sol
#
#
# def drawPhasePortrait3D(args,
#                         deltaX=1, deltaY=1, deltaZ=1,
#                         startX=0, stopX=5,
#                         startY=0, stopY=5,
#                         startZ=0, stopZ=5,
#                         ts=10, nt=101):
#     fig = plt.figure()
#     ax = fig.add_subplot(2, 2, 1, projection='3d')
#     ax.set_title("3D")
#     plt.subplot(2, 2, 2)
#     plt.title("X-Y")
#     plt.grid()
#     plt.subplot(2, 2, 3)
#     plt.title("X-Z")
#     plt.grid()
#     plt.subplot(2, 2, 4)
#     plt.title("Y-Z")
#     plt.grid()
#
#     for x in range(startX, stopX, deltaX):
#         for y in range(startY, stopY, deltaY):
#             for z in range(startZ, stopZ, deltaZ):
#                 sol = calcODE(args, x, y, z, ts, nt)
#
#                 ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
#                 plt.subplot(2, 2, 2)
#                 plt.plot(sol[:, 0], sol[:, 1])
#                 plt.subplot(2, 2, 3)
#                 plt.plot(sol[:, 0], sol[:, 2])
#                 plt.subplot(2, 2, 4)
#                 plt.plot(sol[:, 1], sol[:, 2])
#
#     plt.show()
#
#
# sigma = 60
# r = 28
# b = 8 / 3
# args = (sigma, r, b)
# drawPhasePortrait3D(args,
#                     deltaX=8, deltaY=8, deltaZ=8,
#                     startX=-10, stopX=10,
#                     startY=-10, stopY=10,
#                     startZ=-10, stopZ=10,
#                     ts=0.5, nt=500)
