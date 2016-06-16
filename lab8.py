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
    return y + h * 0.5 * (f(x, y) + f(x + h, y + h * f(x, y)))

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

# Actual value of function in x = b
I = 321.3685907710028004657

# Calculate list of errors for my excellent plots
euler_y = [np.log2(np.abs(get_last_y(a, b, 2**i, euler) - I)) for i in range(1,21)]
euler_mod_y = [np.log2(np.abs(get_last_y(a, b, 2**i, euler_mod) - I)) for i in range(1,21)]
runge3_y = [np.log2(np.abs(get_last_y(a, b, 2**i, rungekutta3) - I)) for i in range(1,21)]
runge4_y = [np.log2(np.abs(get_last_y(a, b, 2**i, rungekutta4) - I)) for i in range(1,21)]

plt.xlabel("iterations")
plt.ylabel("Log2(I* - I)")

plt.plot(np.arange(0, 20), euler_y, color="#bdc3c7")
plt.plot(np.arange(0, 20), euler_mod_y, color="lightblue")
plt.plot(np.arange(0, 20), runge3_y, color="red")
plt.plot(np.arange(0, 20), runge4_y, color="#16a085")

plt.legend(handles=[mp.Patch(color='#bdc3c7', label='Euler'),
                    mp.Patch(color='lightblue', label='Euler mod.'),
                    mp.Patch(color='red', label='R-K 3.'),
                    mp.Patch(color='#16a085', label='R-K 4.')])
plt.show()
