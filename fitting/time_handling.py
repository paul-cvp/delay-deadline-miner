import pm4py
import numpy as np
import pandas as pd
import streamlit as st
from copy import deepcopy
import uuid
from enum import Enum
from pm4py.algo.discovery.dcr_discover.extenstions.time_constraints import TimeMining

from pm4py.algo.discovery.dcr_discover import algorithm as dcr_alg

from fitter import Fitter, get_common_distributions, get_distributions
from pm4py.util.external_functions import *

def time_precision(x):
    match st.session_state.time_granule:
        case "Days",:
            return x.days
        case "Hours":
            return int(x / pd.Timedelta(hours=1))
        case "Minutes":
            return int(x.total_seconds() / 60)
        case "Seconds":
            return x.total_seconds()
        case _:
            return x.days

def remove_rule(rule, e1, e2):
    del st.session_state.all_timings[(rule, e1, e2)]
    del st.session_state.timings_filter_map[(rule, e1, e2)]
    st.session_state.last_change = None
    if rule=='conditions':
        conds = st.session_state.graph.conditions
        if e1 in conds and e2 in conds[e1]:
            conds[e1].remove(e2)
        if len(conds[e1]) == 0:
            del conds[e1]
        st.session_state.graph.conditions = conds
    if rule=='responses':
        resps = st.session_state.graph.responses
        if e1 in resps and e2 in resps[e1]:
            resps[e1].remove(e2)
        if len(resps[e1]) == 0:
            del resps[e1]
        st.session_state.graph.responses = resps


def add_rule(rule, e1, e2):
    tm = TimeMining()
    v1,v2 = tm.get_timings_one_relation(st.session_state.original_log, e1, e2, rule)
    st.session_state.all_timings[(rule, e1, e2)] = v1
    st.session_state.timings_filter_map[(rule, e1, e2)] = v2
    st.session_state.last_change = (rule, e1, e2)
    if rule=='conditions':
        conds = st.session_state.graph.conditions
        if e1 not in conds:
            conds[e1] = {}
        if e2 not in conds[e1]:
            conds[e1].add(e2)
        st.session_state.graph.conditions = conds
    if rule=='responses':
        resps = st.session_state.graph.responses
        if e1 not in resps:
            resps[e1] = {}
        if e2 not in resps[e1]:
            resps[e1].add(e2)
        st.session_state.graph.responses = resps

def last_changed(rule, e1, e2):
    st.session_state.last_change = (rule, e1, e2)

def fit_and_plot():
    xmin = 0
    density = True
    fitted_functions = {}
    rsss = []

    for (rule, e1, e2), timedeltas in st.session_state.all_timings.items():
        if len(timedeltas) == 0:
            print(f"[x] No timings for {e1} {rule} {e2}")
        else:
            global_min_td = min(timedeltas)
            global_max_td = max(timedeltas)
            if f'{e1}{rule}{e2}' in st.session_state:
                curr_min_td, curr_max_td = st.session_state[f'{e1}{rule}{e2}']
                timedeltas_temp = []
                for td in timedeltas:
                    if curr_min_td < td < curr_max_td:
                        timedeltas_temp.append(td)
                timedeltas = timedeltas_temp

            timings = np.asarray([time_precision(x) for x in timedeltas], dtype=int)
            if len(timings) > 0:
                xmax = max(timings)
                Nbins, binwidth = freedman_diaconis_rule(timings)
                counts, bin_edges = np.histogram(timings, bins=Nbins, range=(xmin, xmax), density=density)
                x = (bin_edges[1:][counts > 0] + bin_edges[:-1][counts > 0]) / 2
                y = counts[counts > 0]
                sy = np.sqrt(counts[counts > 0])
                if density:
                    sy = sy * counts[counts > 0]
                fig, ax = plt.subplots(figsize=(12, 6))
                counts, bins, bars = ax.hist(timings, bins=Nbins, range=(xmin, xmax), histtype='step',
                                             density=density, alpha=1, color='g',
                                             label='Binned Duration Data')
                ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k',
                            ecolor='k',
                            elinewidth=1, capsize=1, capthick=1)
                f = Fitter(timings, distributions=get_common_distributions(), xmax=xmax, timeout=2 * 60, bins=Nbins, density=density)
                f.fit()
                # here you get the rss fit scores
                res = f.summary(plot=False)
                residual_sumssquare_error = res.iloc[0].sumsquare_error
                aic = res.iloc[0].aic
                kl_div = res.iloc[0].kl_div
                best_dist, fitted_params = f.get_best().popitem()

                size = 1000
                dist_func = getattr(stats, best_dist)
                start = dist_func.ppf(0.01, **fitted_params)
                end = dist_func.ppf(0.99, **fitted_params)
                x = np.linspace(start, end, size)
                y = dist_func.pdf(x, **fitted_params)
                pdf = pd.Series(y, x)
                pdf.plot(lw=2, label=f'PDF of {best_dist}', color='r', legend=True)

                # ax.plot(f.fitted_pdf[best_dist],label=f'Best fit {best_dist}')
                d = {'Fitter best': f'{best_dist}',
                     'RSS': residual_sumssquare_error,
                     **fitted_params}
                text = nice_string_output(d, extra_spacing=2, decimals=3)
                add_text_to_ax(0.7, 0.6, text, ax, fontsize=14)
                ax.set_xlabel('Duration (Days)')
                ax.set_ylabel('Binned count (Density)')
                ax.set_xlim([xmin - 1, xmax])
                # ax.set_ylim([0, 0.015])

                ax.set_title(f'Timings for "{e1}" {rule} "{e2}"', fontsize=20)
                ax.legend()
                fig.tight_layout()
                # plt.show()
                st.pyplot(fig, clear_figure=True)

                min_td = min(timedeltas)
                max_td = max(timedeltas)
                timedeltas.append(global_min_td)
                timedeltas.append(global_max_td)
                col1, col2 = st.columns(spec=[0.8,0.2],gap="medium")
                with col1:
                    st.select_slider(f"Filter timings for {e1} {rule} {e2}",options=sorted(timedeltas),key=f'{e1}{rule}{e2}', value=(min_td,max_td),on_change=last_changed,args=(rule, e1, e2))
                with col2:
                    st.button("Remove all",key=f'remove_{e1}{rule}{e2}', on_click=remove_rule,args=(rule,e1,e2))
                    st.button('Reset',key=f'reset_{e1}{rule}{e2}', on_click=add_rule,args=(rule,e1,e2))

                # deterministic as in 3.3 from spn delay paper
                if residual_sumssquare_error > 1000:
                    dist_func = stats.mode(timings, keepdims=True)[0][0]
                    fitted_params = {}
                    residual_sumssquare_error = 0

                rsss.append(residual_sumssquare_error)
                fitted_functions[(rule, e1, e2)] = (dist_func, fitted_params)
    return fitted_functions, rsss