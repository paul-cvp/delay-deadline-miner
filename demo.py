from copy import deepcopy

import streamlit as st
import pandas as pd
import numpy as np
import time
import pm4py
import tempfile
import pathlib
from pm4py.algo.discovery.dcr_discover.extenstions.time_constraints import TimeMining
from fitting.time_handling import fit_and_plot, time_precision
from pm4py.visualization.dcr.visualizer import apply as dcr_view

st.markdown("""
Delay Deadline miner Demo \n
You can filter based on the timing data \n
At the end you have the option to add the filtered timing data to you dcr graph and export a timed dcr graph \n
You can also save the subset of the log that you filtered according to the plots \n
""")
st.session_state.strict = False
display = ("Days","Hours","Minutes","Seconds")
st.selectbox("Choose the duration granularity",display,key="time_granule")

uploaded_file = st.file_uploader(label='Upload your event log', type=['xes','xes.gz'])#,on_change=clear_log)
temp_dir = tempfile.TemporaryDirectory()

def save_dcr_graph():
    graph = st.session_state.graph
    graph_png_path = str(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}.png')
    pm4py.save_vis_dcr(graph, graph_png_path, time_precision=st.session_state.time_granule[0])
    graph_xml_path = str(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}.xml')
    pm4py.write_dcr_xml(graph, graph_xml_path)

if uploaded_file is not None:
    if 'log' in st.session_state:
        log = st.session_state.log
        no_cases_before_filter = log['case:concept:name'].nunique()
        st.write(f'Cases in log {no_cases_before_filter}')
        cases = set(log['case:concept:name'].unique())
        # for (rule, e1, e2) in st.session_state.all_timings.keys():
        if st.session_state.last_change:
            (rule,e1,e2) = st.session_state.last_change
            if f'{e1}{rule}{e2}' in st.session_state:
                min_td, max_td = st.session_state[f'{e1}{rule}{e2}']
                fm = deepcopy(st.session_state.timings_filter_map[(rule, e1, e2)])
                for key, value in fm.items():
                    if len(value) < no_cases_before_filter:
                        if min_td > key > max_td:
                            log = log[~log['case:concept:name'].isin(value)]
                            print(f'Removing: {len(value)} cases from {no_cases_before_filter}')
                            del st.session_state.timings_filter_map[(rule, e1, e2)][key]
                    elif key in st.session_state.all_timings[(rule, e1, e2)]:
                        st.session_state.all_timings[(rule, e1, e2)].remove(key)

        graph, _ = pm4py.discover_dcr(log,post_process={'timed'})
        st.graphviz_chart(dcr_view(graph,parameters={'time_precision':st.session_state.time_granule[0]}))

        st.session_state.log = log
        st.session_state.graph = graph

        fitted_functions, rsss_all = fit_and_plot()
    else:
        uploaded_file_name = uploaded_file.name
        st.session_state.log_name = uploaded_file_name.replace('.xes','').replace('.gz','')
        uploaded_file_path = pathlib.Path(temp_dir.name) / uploaded_file_name
        with open(uploaded_file_path, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_file.read())
        log = pm4py.read_xes(uploaded_file_path)
        graph, _ = pm4py.discover_dcr(log,post_process={'timed'})
        st.graphviz_chart(dcr_view(graph,parameters={'time_precision':st.session_state.time_granule[0]}))

        time_mining = TimeMining()
        all_timings, timings_filter_map = time_mining.get_filter_map(log, graph, parameters=None)

        cids = set(log['case:concept:name'].unique())
        st.write(f'Cases in log {len(cids)}')
        all_cases_lt = {}
        all_cases_gt = {}
        for key, value in timings_filter_map.items():
            total_cases_for_rel = set()
            for time, cases_with_time in value.items():
                total_cases_for_rel = total_cases_for_rel.union(cases_with_time)
                if cases_with_time == cids:
                    print(f'[NO!] {time}, {key} <------------------')
                # print(f'Rel: {key} for time {time} has {len(cases_with_time)} cases')
            print(f'[{key}] Adds up to {len(total_cases_for_rel)} which compared to total cases {len(cids)} is Equal={cids==total_cases_for_rel}?')

        st.session_state.timings_filter_map = timings_filter_map
        st.session_state.all_timings = all_timings
        st.session_state.log = log
        st.session_state.graph = graph

        st.session_state.original_cids = cids
        st.session_state.original_timings_filter_map = timings_filter_map
        st.session_state.original_all_timings = all_timings
        st.session_state.original_log = log
        st.session_state.original_graph = graph

        fitted_functions, rsss_all = fit_and_plot()

elif 'log' in st.session_state:
    for key in st.session_state.keys():
        del st.session_state[key]

col1, col2, col3 = st.columns([1,1,1])

def get_filtered_log():
    pm4py.write_xes(st.session_state.log,str(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}_filtered.xes'))

if 'log' in st.session_state:
    with col1:
        get_filtered_log()
        with open(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}_filtered.xes', "rb") as file:
            st.download_button('Save filtered log',file, file_name=f'{st.session_state.log_name}.xes', mime='text/xml')
    with col2:
        save_dcr_graph()
        with open(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}.xml', "rb") as file:
            st.download_button('Save DCR graph as PNG',file, file_name=f'{st.session_state.log_name}.png', mime='image/png')
    with col3:
        save_dcr_graph()
        with open(pathlib.Path(temp_dir.name) / f'{st.session_state.log_name}.png', "rb") as file:
            st.download_button('Export DCR graph as XML',file, file_name=f'{st.session_state.log_name}.xml', mime='text/xml')
