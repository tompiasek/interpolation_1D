from matplotlib import pyplot as plt
import numpy as np


def invert(x):
    return np.sin(np.power(x, -1))


def signum(x):
    return np.sign(np.sin(8*x))


def interpolate(x_arr, y_arr, x_result, kernel):
    """
    Interpolate data using the specified kernel

    Args:
        :param x_arr: The x-values of the original data (function).
        :param y_arr: The y-values of the original data (function).
        :param x_result: The x-values for interpolation.
        :param kernel: The interpolation kernel function

    :return: numpy.ndarray: The interpolated y-values
    """
    if len(x_arr) < 1:
        print("Err: x_arr can't be empty!")
        return 0

    y_result = []
    # range_len = np.abs(x_arr[0] - x_arr[-1])  # Length of measured range
    # distance = range_len / (len(x_result) - 1)  # Distance between two points

    for x in x_result:
        sum_temp = 0
        weight_sum = 0
        for x_compare, y in zip(x_arr, y_arr):
            t = x - x_compare
            weight = kernel(t)
            sum_temp += y * weight
            weight_sum += weight

        if weight_sum != 0:
            y_result.append(sum_temp / weight_sum)
        else:
            y_result.append(0)

    return y_result


def rectangular_kernel(t):
    if np.abs(t) <= 1:
        return 1
    return 0


def h2_kernel(t):
    if -0.5 <= t < 0.5:
        return 1
    return 0


def h3_kernel(t):
    if abs(t) <= 1:
        return 1 - abs(t)
    return 0


def sin_kernel(t):
    if t > 0:
        return np.sin(t) / t
    return 0


def sinc_kernel(t):
    if t > 0:
        return np.sin(np.pi * t) / (np.pi * t)
    return 0


def cubic_kernel(t):
    if 0 < abs(t) < 1:
        return (3 * pow(abs(t), 3)) / 2 - (5 * pow(abs(t), 2)) / 2 + 1
    elif 1 < abs(t) < 2:
        return (-pow(abs(t), 3)) / 2 + (5 * pow(abs(t), 2)) / 2 - (4 * abs(t)) + 2
    return 0


def check_mse(y_original, y_interpolated):
    sum = 0
    k = round(len(y_interpolated) / len(y_original))
    i = 0
    for y in y_original:
        sum += pow((y - y_interpolated[i]), 2)
        i += k

    return sum / len(y_original)


def print_mse(func, x_samples, y_samples, x):
    print(str(func) + " rectangular kernel MSE: " +
          str(check_mse(y_samples, interpolate(x_samples, y_samples, x, rectangular_kernel))))
    print(str(func) + " h2 kernel MSE: " +
          str(check_mse(y_samples, interpolate(x_samples, y_samples, x, h2_kernel))))
    print(str(func) + " kernel MSE: " +
          str(check_mse(y_samples, interpolate(x_samples, y_samples, x, h3_kernel))))
    print(str(func) + " sinusoidal kernel MSE: " +
          str(check_mse(y_samples, interpolate(x_samples, y_samples, x, sin_kernel))))
    print(str(func) + " cubic kernel MSE: " +
          str(check_mse(y_samples, interpolate(x_samples, y_samples, x, cubic_kernel))) + "\n")


def set_subplot_properties(ax, title):
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    # ax.grid(True)


if __name__ == "__main__":
    """######################### SAMPLES PART ############################"""

    no_samples = 1000
    func_density = 200

    x = np.linspace(-np.pi, np.pi, func_density)
    sin_y = np.sin(x)
    inv_y = invert(x)
    sign_y = signum(x)

    x_samples = np.linspace(-np.pi, np.pi, no_samples)
    sin_y_samples = np.sin(x_samples)
    inv_y_samples = invert(x_samples)
    sign_y_samples = signum(x_samples)

    """######################### PLOT PART ################################"""

    bar_width = 2 * np.pi / no_samples

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax_base = ax[0, 0]
    ax_sin = ax[0, 1]
    ax_inv = ax[1, 0]
    ax_sign = ax[1, 1]

    set_subplot_properties(ax_base, "Oryginalne funkcje")
    set_subplot_properties(ax_sin, "f(x)=sin(x)")
    set_subplot_properties(ax_inv, "f(x)=sin(x^(-1))")
    set_subplot_properties(ax_sign, "f(x)=sgn(sin(8x))")

    ax_base.plot(x, sin_y)
    ax_base.plot(x, inv_y)
    ax_base.plot(x, sign_y)

    """SINUS FUNCTION PLOTTING"""
    ax_sin.plot(x, sin_y)
    # ax_sin.bar(x_samples, sin_y_samples, width=bar_width, align='center')
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, rectangular_kernel))
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, h2_kernel))
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, h3_kernel))
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, sin_kernel))
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, cubic_kernel))

    print_mse("Sinus", x_samples, sin_y_samples, x)

    """INVERSE X SINUS PLOTTING"""
    ax_inv.plot(x, inv_y)
    # ax_inv.plot(x_samples, inv_y_samples)
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, rectangular_kernel))
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, h2_kernel))
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, h3_kernel))
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, sin_kernel))
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, cubic_kernel))

    print_mse("Inverse sin", x_samples, inv_y_samples, x)

    """SIGNUM FUNCTION PLOTTING"""
    ax_sign.plot(x, sign_y)
    # ax_sign.plot(x_samples, sign_y_samples)
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, rectangular_kernel))
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, h2_kernel))
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, h3_kernel))
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, sin_kernel))
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, cubic_kernel))

    print_mse("Signum", x_samples, sign_y_samples, x)

    plt.tight_layout()
    plt.show()
