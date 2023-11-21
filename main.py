from matplotlib import pyplot as plt
import numpy as np
from utilities import find_closest, find_closest_indexes, invert, signum
import kernels as krn


def interpolate(x_arr: np.ndarray, y_arr: np.ndarray, x_result: np.ndarray, kernel, interp_range=0):
    """
    Interpolate data using the specified kernel

    Args:
        :param x_arr: The x-values of the original data (function).
        :param y_arr: The y-values of the original data (function).
        :param x_result: The x-values for interpolation.
        :param kernel: The interpolation kernel function
        :param interp_range: The range of points near the interpolated point on which we perform interpolation  # UPDATE

    :return: numpy.ndarray: The interpolated y-values
    """
    if len(x_arr) < 1:
        print("Err: x_arr can't be empty!")
        return 0

    y_result = []
    # range_len = np.abs(x_arr[0] - x_arr[-1])  # Length of measured range
    # distance = range_len / (len(x_result) - 1)  # Distance between two points

    for i in range(len(x_result)):
        if interp_range > 0:
            temp_x_arr = find_closest(x_result[i], x_arr, interp_range)
            temp_y_arr = []
            for index in find_closest_indexes(x_result[i], x_arr, interp_range):
                temp_y_arr.append(y_arr[int(index)])
        else:
            temp_x_arr = x_arr
            temp_y_arr = y_arr

        weights = kernel(x_result[i] - temp_x_arr)
        weights = weights.astype(float)
        total_weight = np.sum(weights)

        if total_weight != 0:
            weights /= total_weight

            y = np.sum(weights * temp_y_arr)
            y_result.append(y)
        else:
            y_result.append(0)

    return y_result


def interpolate_without_normalization(x_arr: np.ndarray, y_arr: np.ndarray, x_result: np.ndarray, kernel, interp_range=0):
    """
    Interpolate data using the specified kernel

    Args:
        :param x_arr: The x-values of the original data (function).
        :param y_arr: The y-values of the original data (function).
        :param x_result: The x-values for interpolation.
        :param kernel: The interpolation kernel function
        :param interp_range: The range of points near the interpolated point on which we perform interpolation  # UPDATE

    :return: numpy.ndarray: The interpolated y-values
    """
    if len(x_arr) < 1:
        print("Err: x_arr can't be empty!")
        return 0

    y_result = []
    # range_len = np.abs(x_arr[0] - x_arr[-1])  # Length of measured range
    # distance = range_len / (len(x_result) - 1)  # Distance between two points

    for i in range(len(x_result)):
        if interp_range > 0:
            temp_x_arr = find_closest(x_result[i], x_arr, interp_range)
            temp_y_arr = []
            for index in find_closest_indexes(x_result[i], x_arr, interp_range):
                temp_y_arr.append(y_arr[int(index)])
        else:
            temp_x_arr = x_arr
            temp_y_arr = y_arr

        weights = kernel(x_result[i] - temp_x_arr)
        weights = weights.astype(float)
        total_weight = np.sum(weights)

        if total_weight != 0:

            y = np.sum(weights * temp_y_arr)
            y_result.append(y)
        else:
            y_result.append(0)

    return y_result


"""######################### MSE ############################"""


def check_mse(y_original, y_interpolated):
    sum = 0
    k = round(len(y_interpolated) / len(y_original))
    i = 0
    for y in y_original:
        sum += pow((y - y_interpolated[i]), 2)
        i += k

    return sum / len(y_original)


def print_mse(func, x_original, y_original, x_interp, interp_range=0):
    print(str(func) + " rectangular kernel MSE: " +
          str(check_mse(y_original, interpolate(x_original, y_original, x_interp, krn.rectangular_kernel, interp_range))))
    print(str(func) + " h2 kernel MSE: " +
          str(check_mse(y_original, interpolate(x_original, y_original, x_interp, krn.h2_kernel, interp_range))))
    print(str(func) + " h3 kernel MSE: " +
          str(check_mse(y_original, interpolate(x_original, y_original, x_interp, krn.h3_kernel, interp_range))))
    print(str(func) + " sinusoidal kernel MSE: " +
          str(check_mse(y_original, interpolate(x_original, y_original, x_interp, krn.sin_kernel, interp_range))))
    print(str(func) + " cubic kernel MSE: " +
          str(check_mse(y_original, interpolate(x_original, y_original, x_interp, krn.cubic_kernel, interp_range))))
    print("\n")


def set_subplot_properties(ax, title):
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    # ax.grid(True)


if __name__ == "__main__":

    """######################### SAMPLES PART ############################"""

    no_samples = 100
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

    ax_base.plot(x, sin_y, label="sin(x)")
    ax_base.plot(x, inv_y, label="sin(x^(-1))")
    ax_base.plot(x, sign_y, label="sgn(sin(8x))")

    ax_base.legend()

    """SINUS FUNCTION PLOTTING"""
    ax_sin.plot(x_samples, sin_y_samples, label="samples")
    # ax_sin.bar(x_samples, sin_y_samples, width=bar_width, align='center')
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, krn.rectangular_kernel), label="rectangular")
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, krn.h2_kernel), label="h2")
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, krn.h3_kernel), label="h3")
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, krn.sin_kernel), label="sinusoidal")
    ax_sin.plot(x, interpolate(x_samples, sin_y_samples, x, krn.cubic_kernel), label="cubic")

    ax_sin.legend()

    print_mse("Sinus", x_samples, sin_y_samples, x, 1)

    """INVERSE X SINUS PLOTTING"""
    ax_inv.plot(x_samples, inv_y_samples, label="samples")
    # ax_inv.plot(x_samples, inv_y_samples)
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, krn.rectangular_kernel), label="rectangular")
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, krn.h2_kernel), label="h2")
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, krn.h3_kernel), label="h3")
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, krn.sin_kernel), label="sinusoidal")
    ax_inv.plot(x, interpolate(x_samples, inv_y_samples, x, krn.cubic_kernel), label="cubic")

    ax_inv.legend()

    print_mse("Inverse sin", x_samples, inv_y_samples, x, 1)

    """SIGNUM FUNCTION PLOTTING"""
    ax_sign.plot(x_samples, sign_y_samples, label="samples")
    # ax_sign.plot(x_samples, sign_y_samples)
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, krn.rectangular_kernel), label="rectangular")
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, krn.h2_kernel), label="h2")
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, krn.h3_kernel), label="h3")
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, krn.sin_kernel), label="sinusoidal")
    ax_sign.plot(x, interpolate(x_samples, sign_y_samples, x, krn.cubic_kernel), label="cubic")

    ax_sign.legend()

    print_mse("Signum", x_samples, sign_y_samples, x, 1)

    """#########################################"""
    fig3, ax3 = plt.subplots(figsize=(12, 8))

    ax3.plot(x_samples, sign_y_samples, label="samples")
    ax3.plot(x, interpolate(x_samples, sign_y_samples, x, krn.rectangular_kernel), label="rectangular (h1)")
    ax3.plot(x, interpolate(x_samples, sign_y_samples, x, krn.h2_kernel), label="h2 kernel")
    ax3.plot(x, interpolate(x_samples, sign_y_samples, x, krn.cubic_kernel), label="cubic")

    set_subplot_properties(ax3, "Porównanie działania jąder interpolacji dla funkcji f(x)=sin(x^(-1)")
    ax3.legend()
    """#########################################"""

    plt.tight_layout()
    plt.show()
