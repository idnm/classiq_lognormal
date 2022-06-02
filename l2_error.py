from scipy.stats import lognorm
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from jax import vmap, jit, value_and_grad, lax
import jax.numpy as jnp
from jax.scipy.special import erf

from qiskit import QuantumCircuit


# Error computation by QuantumSage

TARGET_ERR = 1e-2


def l2_error(pmf: np.array, x_grid: np.array, sigma=0.1):
    pmf = np.array(pmf)
    x_grid = np.array(x_grid)
    assert all(pmf >= 0)
    assert np.isclose(sum(pmf), 1)
    assert all(x_grid >= 0)
    assert all(np.diff(x_grid) > 0)
    assert len(pmf) + 1 == len(x_grid)

    n_point = 2 ** 22
    tail_value = (TARGET_ERR / 100) ** 2
    min_x = lognorm.ppf(tail_value, sigma)
    max_x = lognorm.ppf(1 - tail_value, sigma)
    x_middle = np.linspace(min_x,max_x, n_point)
    x_lower_tail = np.linspace(0, min_x, n_point//1000)
    x_upper_tail = np.linspace(max_x, x_grid[-1], n_point//1000) if x_grid[-1] > max_x else np.array([])


    x_approx = np.diff(x_grid) / 2 + x_grid[:-1]
    x_approx = np.concatenate(([x_grid[0]], x_approx, [x_grid[-1]]))
    pdf_approx = pmf /np.diff(x_grid)
    pdf_approx = np.concatenate(([pdf_approx[0]], pdf_approx, [pdf_approx[-1]]))

    fy = interp1d(x_approx, pdf_approx, kind='nearest', assume_sorted=True, fill_value=(0, 0), bounds_error=False)
    x_full = np.concatenate((x_lower_tail[:-1], x_middle, x_upper_tail[1:]))
    approx_pdf = fy(x_full)

    full_pdf = lognorm.pdf(x_full, sigma)
    dx = np.diff(x_full)
    dx = np.append(dx, 0)

    plt.plot(full_pdf)
    plt.plot(approx_pdf)

    upper_tail_err_2_approx = lognorm.sf(x_full[-1], sigma)
    main_err_2 = sum((full_pdf - approx_pdf) ** 2 * dx)
    err = (upper_tail_err_2_approx + main_err_2) ** 0.5
    return err


def vanilla():
    s = 0.1
    qubits = 12
    partial = 2 ** qubits
    tail_value = (TARGET_ERR / 100) ** 2
    x_approx = np.linspace(lognorm.ppf(tail_value, s),
    lognorm.ppf(1 - tail_value, s), partial + 1)
    pmf = np.diff(lognorm.cdf(x_approx, s))
    pmf = pmf / sum(pmf)
    print(l2_error(pmf, x_approx, sigma=s))


# Error computation by tnemoz
def tnemoz_l2_error(p, x):
    p = np.array(p)
    x = np.array(x)
    assert all(p >= 0)
    assert np.isclose(sum(p), 1)
    assert all(x >= 0)
    assert all(np.diff(x) > 0)
    assert len(p) + 1 == len(x)
    err_squared = quad(lambda x: lognorm.pdf(x, s=0.1) ** 2, 0, float("inf"))[0]
    diff = x[1:] - x[:-1]
    err_squared += np.sum((p ** 2) / diff)
    temp = np.array([quad(lambda x: lognorm.pdf(x, s=0.1), x[i], x[i + 1])[0] for i in range(x.shape[0] - 1)])
    err_squared -= 2 * np.sum(p * temp / diff)

    return np.sqrt(err_squared)

# Error computation by idnm


def lognormal(x):
    s = 0.1
    mu = 0

    return 1 / (jnp.sqrt(2 * jnp.pi) * x * s) * jnp.exp(-(jnp.log(x) - mu) ** 2 / 2 / s ** 2)


def lognormal_int(x):
    return erf(5 * jnp.sqrt(2) * jnp.log(x)) / 2


def lognormal_squared_int(x):
    return erf(10 * jnp.log(x) + 1 / 20) / 2 / jnp.sqrt(jnp.pi) * 5 * jnp.power(jnp.e, 1 / 400)


def idnm_l2_error(probs, inner_grid, left=0, right=jnp.inf):
    values = probs / (inner_grid[1:]-inner_grid[:-1])
    return jnp.sqrt(l2_error_contributions(values, inner_grid, left, right).sum())


def l2_error_contributions(values, inner_grid, left, right):
    # inner contributions
    f_squared_contrib = vmap(lognormal_squared_int)(inner_grid[1:]) - vmap(lognormal_squared_int)(inner_grid[:-1])
    f_contrib = vmap(lognormal_int)(inner_grid[1:]) - vmap(lognormal_int)(inner_grid[:-1])
    const_contrib = (values ** 2) * (inner_grid[1:] - inner_grid[:-1])

    # outer contributions
    outer_contrib_left = lognormal_squared_int(inner_grid[0]) - lognormal_squared_int(left)
    outer_contrib_right = lognormal_squared_int(right) - lognormal_squared_int(inner_grid[-1])

    # total
    total_contribs = f_squared_contrib - 2 * f_contrib * values + const_contrib + outer_contrib_left + outer_contrib_right

    return total_contribs