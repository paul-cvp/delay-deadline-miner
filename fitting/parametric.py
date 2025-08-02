import os
import numpy as np
import matplotlib.pyplot as plt

from fitter import Fitter, get_common_distributions
from distfit import distfit
from iminuit import Minuit,cost
from scipy import stats
from discover.util import freedman_diaconis_rule

from fitting.external_functions import *
from discover.simulation import Simulation

from fitting.functions import Functions

class Parametric:

    sim = Simulation()

    def simple_distribution_fit_all_timings(self,timings,folder, xmax=1000):
        res = {}
        for (rule, e1, e2), data in timings.items():
            if len(data) > 5:  # too low statistics, needs at least 5 data points (or another limit).
                file_name = os.path.join(folder, f'{rule}-{e1}-{e2}_simple_fit.jpg')
                plot_title = f'{rule}, {e1} --> {e2}'
                res[(rule, e1, e2)] = self.simple_distribution_fitter(data, file_name, plot_title, None, xmax)
            else:
                print(f'[!] NOT Enough Data Points for fitting {rule}: {e1}-->{e2}')
                res[(rule, e1, e2)] = None
        return res

    def simple_distribution_fitter(self,data,filename,title,Nbins=None,xmax=None,save=False):
        '''
        :param data:
        :param filename:
        :param Nbins:
        :param xmax:
        :return: name of the best fitting distribution
        '''
        if not Nbins:
            Nbins, bin_width = freedman_diaconis_rule(data)
        f = Fitter(data, distributions=get_common_distributions(), xmax=xmax, timeout=2 * 60, bins=Nbins)
        f.fit()
        if save:
            fig, ax = plt.subplots(figsize=(16, 5))
            f.summary(plot=True)
            fig.tight_layout()
            ax.set_xlabel('Duration (Days)')
            ax.set_ylabel('Binned count')
            ax.set_title(title)
            plt.savefig(filename)
            plt.close()
        return f.get_best()#.summary().head(1).index[0]

    def simple_distribution_distfit(self,data,filename,title,Nbins=None,xmax=1000):
        '''
        :param data:
        :param filename:
        :param Nbins:
        :param xmax:
        :return: name of the best fitting distribution
        '''
        if not Nbins:
            Nbins, bin_width = freedman_diaconis_rule(data)
        dist = distfit(distr='popular')  # full, popular, or a specific name
        dist.fit_transform(data)
        fig, ax = plt.subplots(figsize=(16, 5))
        dist.plot()
        fig.tight_layout()
        ax.set_xlabel('Duration (Days)')
        ax.set_ylabel('Binned count')
        ax.set_title(title)
        plt.savefig(filename)
        plt.close()
        return dist.summary.iloc[0]['distr']

    def old_advanced_distribution_fit(self,data):
        '''
        #TODO: DOUBLE CHECK FITTING AND BIN WIDTH
            * select function to fit
                * initialize initial_values_dict
                * fit the binned data with fit_minuit
                * check the chi2 score and update initial_values_dict (use a search space, heuristic etc)
                * find the parameters that minimize the chi2, stop and save
            * select another function to fit and repeat
                * compare all fitted functions save the one with the lowest chi2
                * save the best plot, the function and the parameters
                * optionally uniformly sample points

        :param data:
        :return:
        '''
        Nbins, bin_width = freedman_diaconis_rule(data)
        f = Functions(bin_width)
        return None

    def advanced_distribution_fit_all_timings(self,timings,functions_to_fit,initial_values_dicts,folder, xmin=0, xmax=1000,save_plot=True):
        res = {}
        for (rule, e1, e2), data in timings.items():
            if len(data) > 5:  # too low statistics, needs at least 5 data points (or another limit).
                key = (rule, e1, e2)
                file_name = os.path.join(folder, f'{rule}-{e1}-{e2}_advanced_fit.jpg')
                function_to_fit = functions_to_fit[key]
                initial_values_dict = initial_values_dicts[key]
                Nbins, binwidth = freedman_diaconis_rule(data)
                if function_to_fit:
                    if isinstance(function_to_fit,int):
                        res[(rule, e1, e2)] = function_to_fit
                    else:
                        initial_values_dict['binwidth'] = binwidth
                        # data for chi2 fit
                        counts, bin_edges = np.histogram(data, bins=Nbins, range=(xmin, xmax))
                        # take only non empty bins, that's why counts>0
                        x = (bin_edges[1:][counts > 0] + bin_edges[:-1][counts > 0]) / 2
                        y = counts[counts > 0]
                        sy = np.sqrt(counts[counts > 0])  # NOTE: We (naturally) assume that the bin count is Poisson distributed.

                        #minuit_fit, fit_info = \
                        minuit_fit = self.fit_minuit(function_to_fit,initial_values_dict,x,y,sy,file_name, xmax)
                        res[(rule, e1, e2)] = (minuit_fit)#, fit_info)

                        if save_plot:
                            fig, ax = plt.subplots(figsize=(16, 5))
                            counts, bins, bars = ax.hist(data, bins=Nbins, range=(xmin, xmax), histtype='step',
                                                         density=False, alpha=1, color='g', label='Binned Duration Data')
                            ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k', ecolor='k',
                                        elinewidth=1, capsize=1, capthick=1)
                            pred_y = []
                            for k in x:
                                pred_y.append(function_to_fit(k, *minuit_fit.values[:]))
                            pred_y = np.array(pred_y)
                            ax.plot(x, pred_y, '-r', label='chi2 fit')  # Note how we can "smartly" input the fit values!
                            d = {
                                'Rule': rule,
                                'Event from': e1,
                                'Event to': e2,
                            }
                            to_print = {}
                            pm = u"\u00B1"
                            for k, v in minuit_fit.values.to_dict().items():
                                to_print[k] = f'{v:.2f}' + pm + f'{minuit_fit.errors.to_dict()[k]:.2f}'
                            #d = {**d, **fit_info}
                            d = {**d, **to_print}
                            text = nice_string_output(d, extra_spacing=2, decimals=3)
                            add_text_to_ax(0.50, 0.77, text, ax, fontsize=14)

                            x_sampled, y_sampled = self.sim.sample_timing_points(function_to_fit, minuit_fit.values[:], 10000, max(counts),xmin, xmax)
                            ax.scatter(x_sampled, y_sampled, s=.1, label='Scatter plot of data')
                            title = f'{rule}, {e1} --> {e2}'
                            ax.set_xlabel('Duration (Days)')
                            ax.set_ylabel('Binned count')
                            ax.set_title(title)
                            ax.legend()
                            fig.tight_layout()
                            plt.savefig(file_name)
                            plt.close()
            else:
                res[(rule, e1, e2)] = None
        return res

    def fit_minuit(self, function_to_fit, initial_values_dict, x, y, sy,file_name,xmax, err_def=1):
        """Fit any defined function using Chi2 or likelihood fit
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
        ls_fit = cost.LeastSquares(x, y, sy, function_to_fit)
        minuit_fit = Minuit(ls_fit, **initial_values_dict)
        minuit_fit.migrad()  # Perform the actual fit
        return minuit_fit
        #TODO make a print

        #Minuit.print_level = 1
        #chi2_fit = Chi2Regression(function_to_fit, x, y, sy)
        #chi2_fit.errordef = 1
        #minuit_fit = Minuit(chi2_fit, **initial_values_dict)
        #minuit_fit.migrad()  # Perform the actual fit

        #Chi2_fit = minuit_fit.fval
        #Ndof_fit = len(x) - len(initial_values_dict)
        #Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)
        #Reduced_chi2 = minuit_fit.fval / (len(x) - minuit_fit.nfit)  # should be roughly 1 for a good fit
        #return (minuit_fit, None)#{'Chi2': Chi2_fit, 'Ndof': Ndof_fit, 'Prob': Prob_fit, 'Reduced chi2': Reduced_chi2})
