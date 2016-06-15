import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mp


def f(x, y):
    """Function f(x, y) """
    return np.e**x * (x+1)**2 + y*2/(x+1)


def euler(x, y, h):
    """ Calculate next y point from
    previous points x and y with Euler method
    :param x: float previous x point
    :param y: float previous y point
    :param h: float step point
    :return: float y value
    """
    return y + h * f(x, y)


def euler_mod(x, y, h):
    """Calculate next y point from
    previous points x and y with modificated Euler method
    :param x: float previous x point
    :param y: float previous y point
    :param h: float step point
    :return: float y value
    """
    y_half = y + 0.5*h * f(x, y)
    return y + h * f(x + h*0.5, y_half)


def rungekutta_p2(x, y, h):
    """Calculate next y point from
    previous points x and y with Runga-Kutta second order method
    :param x: float previous x point
    :param y: float previous y point
    :param h: float step
    :return: float y value
    """
    return y + h*0.5 * (f(x, y) + f(x + h, y + h*f(x, y)))


def rungekutta3(x, y, h):
    """Calculate next y point from
    previous points x and y with Runga-Kutta third order method
    :param x: float previous x point
    :param y: float previous y point
    :param h: float step
    :return: float y value
    """
    k1 = f(x, y)
    k2 = f(x + h*0.5, y + k1*0.5*h)
    k3 = f(x + h, y + 2*k2*h - k1 * h)
    return y + (k1 + 4*k2 + k3) / 6 * h


def rungekutta4(x, y, h):
    """Calculate next y point from
    previous points x and y with Runga-Kutta fourth order method
    :param x: float previous x point
    :param y: float previous y point
    :param h: float step
    :return: float y value
    """
    k1 = f(x, y)
    k2 = f(x + h / 4, y + k1 / 4 * h)
    k3 = f(x + h / 2, y + k2 / 2 * h)
    k4 = f(x + h, y + k1 * h - 2 * k2 * h + 2 * k3 * h)
    return y + (k1 + 4 * k3 + k4) / 6 * h


def get_last_y(a, b, n, method):
    """Calculate y value on the b point according to given method.
    :param a: left border (float)
    :param b: right border (float)
    :param n: number of partitions (int)
    :param method: function with method
    :return:
    """
    y = 1   # From the Koshi condition
    x = a
    h = (b - a) / float(n)

    for i in range(n):
        y = method(x, y, h)
        x += h
    return y

# Borders [a, b]
a = 0
b = 3

# storage for errors
error_euler = []
error_euler_mod = []
error_runge2 = []
error_runge3 = []
error_runge4 = []

# Actual value of function in x = b
I = 321.3685907710028004657

# Calculate list of errors for my excellent plots
for i in range(1, 21):
    # get y values at the end of the segment
    euler_y = get_last_y(a, b, 2**i, euler)
    euler_mod_y = get_last_y(a, b, 2**i, euler_mod)
    runge2_y = get_last_y(a, b, 2**i, rungekutta_p2)
    runge3_y = get_last_y(a, b, 2**i, rungekutta3)
    runge4_y = get_last_y(a, b, 2**i, rungekutta4)

    # get errors for plots
    error_euler.append(np.log2(np.abs(I - euler_y)))
    error_euler_mod.append(np.log2(np.abs(I - euler_mod_y)))
    error_runge2.append(np.log2(np.abs(I - runge2_y)))
    error_runge3.append(np.log2(np.abs(I - runge3_y)))
    error_runge4.append(np.log2(np.abs(I - runge4_y)))

# Plots

# Euler
plt.plot(np.arange(0, 20), error_euler, color="#bdc3c7")
gray_patch = mp.Patch(color='#bdc3c7', label='Euler')

# Euler with modification
plt.plot(np.arange(0, 20), error_euler_mod, color="lightblue")
blue_patch = mp.Patch(color='lightblue', label='Euler mod.')

# Runge-Kutta 2 order
plt.plot(np.arange(0, 20), error_runge2, color="pink")
pink_patch = mp.Patch(color='pink', label='R-K 2.')

# Runge-Kutta 3 order
plt.plot(np.arange(0, 20), error_runge3, color="red")
green_patch = mp.Patch(color='red', label='R-K 3.')

# Runge-Kutta 4 order
plt.plot(np.arange(0, 20), error_runge4, color="#16a085")
mag_patch = mp.Patch(color='#16a085', label='R-K 4.')

plt.legend(handles=[gray_patch, blue_patch, pink_patch, green_patch, mag_patch])
plt.show()
