
import numpy as np
from iminuit import Minuit,cost                             # The actual fitting tool, better than scipy's
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# =============================================================================
#  Probfit replacement
# =============================================================================

from iminuit.util import make_func_code
from iminuit import describe  # , Minuit,


r = np.random                         # Random generator
r.seed(1)

# Author: Christian Michelsen, NBI, 2018
# Extended by Paul Cosma, DIKU, 2022

def plot_hist_of_exp_data(tau=1 / np.e):
    # General input:
    Nbins = 100
    xmin, xmax = 0, 10
    binwidth = (xmax - xmin) / Nbins
    Nbkg = 5000  # Number of random Exponential points
    x_bkg = r.exponential(tau, Nbkg)
    x_all = x_bkg
    # Create just a single figure and axes, and a (classic) histogram:
    fig, ax = plt.subplots(figsize=(16, 6))  # figsize is in inches
    hist = ax.hist(x_all, bins=Nbins, range=(xmin, xmax), histtype='step', linewidth=2, color='red', label='Data, normal histogram')

    # Find the x, y and error on y (sy) given the histogram:
    counts, bin_edges = np.histogram(x_all, bins=Nbins, range=(xmin, xmax))
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    y = counts
    sy = np.sqrt(counts)  # NOTE: We (naturally) assume that the bin count is Poisson distributed.
    # This is an approximation, since there is a low count in the last bins.

    # Did you make sure, that all bins were non-zero???
    # x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    # y = counts[counts>0]
    # sy = np.sqrt(counts[counts>0])   # NOTE: We (naturally) assume that the bin count is Poisson distributed.

    # Now create a histogram with uncertainties (better, I would argue):
    ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k', ecolor='k', elinewidth=1, capsize=1, capthick=1)

    # Set the figure texts; xlabel, ylabel and title.
    ax.set(xlabel="Random numbers",  # the label of the y axis
           ylabel="Frequency / 10",  # the label of the y axis
           title="Distribution of Gaussian and exponential numbers")  # the title of the plot
    ax.legend(loc='best', fontsize=20);  # could also be # loc = 'upper right' e.g.


def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.floating)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None


def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else:
        return np.ones_like(x)


def compute_f(f, x, *par):
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one

    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x = x[mask]
            y = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)

        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters

        # compute the function value
        f = compute_f(self.f, self.x, *par)

        # compute the chi2-value
        chi2 = np.sum(self.weights * (self.y - f) ** 2 / self.sy ** 2)

        return chi2


def simpson38(f, edges, bw, *arg):
    yedges = f(edges, *arg)
    left38 = f((2. * edges[1:] + edges[:-1]) / 3., *arg)
    right38 = f((edges[1:] + 2. * edges[:-1]) / 3., *arg)

    return bw / 8. * (np.sum(yedges) * 2. + np.sum(left38 + right38) * 3. - (yedges[0] + yedges[-1]))  # simpson3/8


def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint + 1)
    bw = edges[1] - edges[0]

    return simpson38(f, edges, bw, *arg)


class UnbinnedLH:  # override the class with a better one

    def __init__(self, f, data, weights=None, bound=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):

        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None):
                weights = weights[mask]

        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue

        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters

        logf = np.zeros_like(self.data)

        # compute the function value
        f = compute_f(self.f, self.data, *par)

        # find where the PDF is 0 or negative (unphysical)
        mask_f_positive = (f > 0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive]

        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value

        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)

        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term

        return llh

    def default_errordef(self):
        return 0.5


class BinnedLH:  # override the class with a better one

    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):

        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None):
                weights = weights[mask]
            if (weighterrors is not None):
                weighterrors = weighterrors[mask]

        self.weights = set_var_if_None(weights, data)

        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None:
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)

        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights ** 2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors ** 2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)

        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv

        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)

        return ret

    def default_errordef(self):
        return 0.5


def xlogyx(x, y):
    # compute x*log(y/x) to a good precision especially when y~x

    if x < 1e-100:
        warnings.warn('x is really small return 0')
        return 0.

    if x < y:
        return x * np.log1p((y - x) / x)
    else:
        return -x * np.log1p((x - y) / y)


# compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x < 1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x < y:
        return w * np.log1p((y - x) / x)
    else:
        return -w * np.log1p((x - y) / y)


def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    N = np.sum(h)
    n = len(edges)

    ret = 0.

    for i in range(n - 1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i + 1]), nint_subdiv, *par)

        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm * N) + (th - tm * N)

            else:
                if w2[i] < 1e-200:
                    continue
                tw = w2[i]
                factor = th / tw
                ret -= factor * (wlogyx(th, tm * N, th) + (th - tm * N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th, tm) + (th - tm)
            else:
                if w2[i] < 1e-200:
                    continue
                tw = w2[i]
                factor = th / tw
                ret -= factor * (wlogyx(th, tm, th) + (th - tm))

    return ret


def compute_bin_lh_f(f, edges, h, w2, extended, use_sumw2, badvalue, *par):
    mask_positive = (h > 0)

    N = np.sum(h)
    midpoints = (edges[:-1] + edges[1:]) / 2
    b = np.diff(edges)

    midpoints_pos = midpoints[mask_positive]
    b_pos = b[mask_positive]
    h_pos = h[mask_positive]

    if use_sumw2:
        warnings.warn('use_sumw2 = True: is not yet implemented, assume False ')
        s = np.ones_like(midpoints_pos)
        pass
    else:
        s = np.ones_like(midpoints_pos)

    E_pos = f(midpoints_pos, *par) * b_pos
    if not extended:
        E_pos = E_pos * N

    E_pos[E_pos < 0] = badvalue

    ans = -np.sum(s * (h_pos * np.log(E_pos / h_pos) + (h_pos - E_pos)))

    return ans


class MinuitFitFunctions:

    def __init__(self,binwidth):
        self.binwidth = binwidth

    @staticmethod
    def freedman_diaconis_rule(data):
        """rule to find the bin width and number of bins from data"""
        if (stats.iqr(data) > 0):
            bin_width = 2 * stats.iqr(data) / len(data) ** (1 / 3)
            Nbins = int(np.ceil((data.max() - data.min()) / bin_width))
            return Nbins, bin_width
        else:
            return 100, 0


    def uniform_pdf(self, x, a, b):
        if a <= x and x <= b:
            return 1 / (b - a)
        else:
            return 0


    def exp_pdf(self, x, tau):
        """Exponential with lifetime tau"""
        return 1.0 / tau * np.exp(-x / tau)


    def gauss_pdf(self, x, mu, sigma):
        """Gaussian"""
        return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


    def N_gauss_pdf(self, x, N, mu, sigma):
        """Gaussian"""
        return N * self.binwidth * self.gauss_pdf(x, mu, sigma)


    def log_gauss_pdf(self, x, mu, sigma):
        return 1.0 / np.sqrt(2 * np.pi) / (sigma * x) * np.exp(-0.5 * (np.log(x) - mu) ** 2 / sigma ** 2)


    def N_log_gauss_pdf(self, x, N, mu, sigma):
        return N * self.binwidth * self.log_gauss_pdf(x, mu, sigma)


    def double_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2):
        return N * self.binwidth * self.gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.gauss_pdf(x, mu2, sigma2)


    def triple_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3):
        return N * self.binwidth * self.gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.gauss_pdf(x, mu2, sigma2) + N3 * self.binwidth * self.gauss_pdf(x, mu3, sigma3)


    def triple_gauss_gauss_log(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3):
        return N * self.binwidth * self.gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.gauss_pdf(x, mu2, sigma2) + N3 * self.binwidth * self.log_gauss_pdf(x, mu3, sigma3)


    # def triple_gauss_log_log(x,N,mu,sigma,N2,mu2,sigma2,N3,mu3,sigma3):
    #     return N*binwidth*gauss_pdf(x,mu,sigma) + N2*binwidth*log_gauss_pdf(x,mu2,sigma2) + N3*binwidth*log_gauss_pdf(x,mu3,sigma3)

    def triple_gauss_log_log(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3):
        if x < 61:
            return N * self.binwidth * self.gauss_pdf(x, mu, sigma)
        else:
            return N2 * self.binwidth * self.log_gauss_pdf(x, mu2, sigma2) + N3 * self.binwidth * self.log_gauss_pdf(x, mu3, sigma3)


    def double_log_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2, binwidth):
        return N * binwidth * self.log_gauss_pdf(x, mu, sigma) + N2 * binwidth * self.log_gauss_pdf(x, mu2, sigma2)


    def double_log_gaussian_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2):
        return N_exp * self.binwidth * self.exp_pdf(x, tau) + N * self.binwidth * self.log_gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.log_gauss_pdf(x, mu2, sigma2)


    def double_gaussian_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2):
        return N_exp * self.binwidth * self.exp_pdf(x, tau) + N * self.binwidth * self.gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.gauss_pdf(x, mu2, sigma2)


    def gaus_log_gauss_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2):
        return N_exp * self.binwidth * self.exp_pdf(x, tau) + N * self.binwidth * self.gauss_pdf(x, mu, sigma) + N2 * self.binwidth * self.log_gauss_pdf(x, mu2, sigma2)


    def fit_minuit(self, function_to_fit, initial_values_dict, x, y, sy, err_def=1):
        """Fit any defined function using least squares
        NOTE: All fixed parameters that are part of the 'function_to_fit'
        need to be initialized before this method call

        Parameters:
        function_to_fit: the function to fit the data to can be anything (gaussian, exp, exp+gauss,a+b)
        initial_values_dict: a dictionary of key value pairs that are the initial input to the function_to_fit
        x: input training data
        y: result training data
        sy: errors on the output (if data is binned errors are poisson)
        err_def: 1: ls fit 0.5: likelihood fit

        returns: Chi2, Ndof, Prob
        """
        Minuit.print_level = 1
        ls_fit = cost.LeastSquares(x, y, sy, function_to_fit)

        minuit_fit = Minuit(ls_fit, **initial_values_dict)
        if 'binwidth' in initial_values_dict.keys():
            minuit_fit.fixed['binwidth'] = True
        minuit_fit.migrad()  # Perform the actual fit

        ls_fit = minuit_fit.fval
        return minuit_fit


    def fit_chi2_minuit(self, function_to_fit, initial_values_dict, x, y, sy, err_def=1):
        """Fit any defined function using Chi2
        NOTE: All fixed parameters that are part of the 'function_to_fit'
        need to be initialized before this method call

        Parameters:
        function_to_fit: the function to fit the data to can be anything (gaussian, exp, exp+gauss,a+b)
        initial_values_dict: a dictionary of key value pairs that are the initial input to the function_to_fit
        x: input training data
        y: result training data
        sy: errors on the output (if data is binned errors are poisson)
        err_def: 1: chi2 fit 0.5: likelihood fit

        returns: Chi2, Ndof, Prob
        """
        Minuit.print_level = 1
        chi2_fit = Chi2Regression(function_to_fit, x, y, sy)
        chi2_fit.errordef = 1
        minuit_fit = Minuit(chi2_fit, **initial_values_dict)
        minuit_fit.migrad()  # Perform the actual fit

        Chi2_fit = minuit_fit.fval
        Ndof_fit = len(x) - minuit_fit.nfit #len(initial_values_dict)
        Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)
        Reduced_chi2 = minuit_fit.fval / Ndof_fit #(len(x) - minuit_fit.nfit)  # should be roughly 1 for a good fit
        return minuit_fit, {'Chi2': Chi2_fit, 'Ndof': Ndof_fit, 'Prob': Prob_fit, 'Reduced chi2': Reduced_chi2}

    def sample_points(self,func_to_fit, func_params, N_samples, max_sample_y, xmin, xmax):
        x_sampled = np.zeros(N_samples)
        y_sampled = np.zeros(N_samples)
        Ntry = 0
        for i in range(N_samples):
            while True:
                Ntry += 1
                sample_y = r.uniform(0, max_sample_y)
                sample_x = r.uniform(xmin, xmax)
                if sample_y < func_to_fit(sample_x, *func_params):
                    break
            x_sampled[i] = sample_x
            y_sampled[i] = sample_y
        return x_sampled, y_sampled