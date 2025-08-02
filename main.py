import sys, getopt, os
import pm4py
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter

from discover.discover import Discover
from discover.timings import Timings
from fitting.parametric import Parametric
from fitting.functions import Functions
from discover.util import *

def load_log(log_path):
    log = None
    if os.path.exists(log_path) and os.path.isfile(log_path):
        if str.endswith(log_path,'.csv'):
            # convert from csv to xes
            log = log_converter.apply(log_path, variant=log_converter.Variants.TO_EVENT_LOG)

        elif str.endswith(log_path,'.xes'):
            # load using pm4py
            log = pm4py.read_xes(log_path)

        return log
    else:
        raise Exception(f'[x] Error, path not found! {log_path}')

def mine_dcr_from_path(log_path, graph_path=None):
    print(f'[i] Mining started')
    disc = Discover()
    log = None
    if os.path.exists(log_path) and os.path.isfile(log_path):
        if graph_path is None:
            graph_path = os.path.splitext(log_path[0])+'.txt'
        if str.endswith(log_path,'.csv'):
            # convert from csv to xes
            log = log_converter.apply(log_path, variant=log_converter.Variants.TO_EVENT_LOG)

        elif str.endswith(log_path,'.xes'):
            # load using pm4py
            log = pm4py.read_xes(log_path)

        status = disc.mine(log, graph_path)
        if status != 0:
            print(f'[x] Some error occured, status={status}')
        else:
            print(f'[i] Succes!')
    else:
        print(f'[x] Error, path not found!')

    print(f'[i] Mining finished')
    return disc

def mine_dcr(disc,log, graph_path=None,findAdditionalConditions=True):
    print(f'[i] Mining dcr graph started')
    if graph_path is not None and os.path.exists(os.path.dirname(os.path.abspath(graph_path))):
        status = disc.mine(log, graph_path,findAdditionalConditions=findAdditionalConditions)
        if status != 0:
            print(f'[x] Some error occured, status={status}')
        else:
            print(f'[i] Succes!')
    else:
        print(f'[x] Error, path not found!')

    print(f'[i] Mining dcr graph finished')
    return disc

def mine_timings(log, graph_path):
    timings = Timings()
    print(f'[i] Mining timings started')
    timing_input_dict = timings.create_timing_input_dict(graph_path)
    res = timings.get_timings(log,timing_input_dict)
    print(f'[i] Mining timings finished')
    return res

def run_for_one_log(log_path,graph_path,results_folder,specific_pairs_to_check=None,known_pairs=None):
    disc = Discover()
    log = load_log(log_path)

    # Mine the graph using discover
    mine_dcr(disc,log,graph_path)
    #TODO: Get events from a BPMN model and their relations + a specification whether it's mining a delay or a deadline
    #extended bpmn diagrams

    #TODO: Get events from a user specified list of events + a specification whether it's mining a delay or a deadline


    timings_raw_data = mine_timings(log,graph_path) # timings is a dict with a tuple for the rule and all the deltas in days (maybe change to seconds in resolution if some other dataset is used)
    # timings into boxplots
    print(f'[i] Started creating boxplot data')
    boxplot_values = create_timing_box_plots(timings_raw_data,results_folder)
    print(f'[i] Finished creating boxplot data')
    #(lower_whisker, lower_quartile, median, upper_quartile, upper_whisker, iqr)
    print(f'[i] Started writing median values as delays and deadlines to DCR graph')
    to_print_values = {}
    to_print_values_outliers = {}
    total_conditions = 0
    total_responses = 0
    total_not_enough_values = 0
    for (k,v) in boxplot_values.items():
        if v:
            if k[0] == 'CONDITION': #min for condition
                to_print_values[k] = v[6]
                to_print_values_outliers[k] = v[0]
                total_conditions = total_conditions + 1
            elif k[0] == 'RESPONSE': #max for response
                to_print_values[k] = v[7]
                to_print_values_outliers[k] = v[4]
                total_responses = total_responses + 1
        else:
            total_not_enough_values = total_not_enough_values + 1
    disc.writeGraph(f'{graph_path}.txt',to_print_values)
    disc.write_with_do_subprocesses(f'{graph_path}_do_subprocesses.txt',to_print_values)
    disc.writeGraph(f'{graph_path}_no_outliers.txt',to_print_values_outliers)

    mean_values = get_mean_values(timings_raw_data)
    print(f'[i] Finished writing values as delays and deadlines to DCR graph')
    # timings into histograms
    print(f'[i] Started creating histograms')
    histogram_values = create_histograms(timings_raw_data,results_folder)

    zero_bins = {'CONDITION':0,'RESPONSE':0}
    single_bins = {'CONDITION':0,'RESPONSE':0}
    multiple_bins = {'CONDITION':0,'RESPONSE':0}
    for histv,histe in histogram_values.items():
        (r, e1, e2) = histv
        if histe:
            (counts, bin_edges) = histe
            v = len(counts[counts>0])
            if v==1:
                single_bins[r] = single_bins[r] + 1
            elif v>1:
                multiple_bins[r] = multiple_bins[r] + 1
            else: # 0 bins
                print(f'[!] {k} ZERO bins?! {v}')
                zero_bins[r] = zero_bins[r] + 1
        else:
            print(f'[!] {k} ZERO bins?! {v}')
            zero_bins[r] = zero_bins[r] + 1


    print(f'[i] Finished creating histograms')
    # histograms + overlay of simple distributions fitted
    fitting_parametric_dist = Parametric()
    #once a list of best fitting functions is provided fit them with iminuit and chi2 to get a nice output.
    print(f'[i] Started fitting simple distributions')
    single_distribution_fits = fitting_parametric_dist.simple_distribution_fit_all_timings(timings_raw_data, results_folder)
    best_dist_counts = {'CONDITION': {},'RESPONSE': {}}
    not_enough_data_counts = {'CONDITION':0,'RESPONSE':0}
    for (r,e1,e2),fit in single_distribution_fits.items():
        if fit:
            best_dist = next(iter(fit))
            if best_dist in best_dist_counts[r].keys():
                best_dist_counts[r][best_dist] = best_dist_counts[r][best_dist] + 1
            else:
                best_dist_counts[r][best_dist] = 1
        else:
            not_enough_data_counts[r] = not_enough_data_counts[r] + 1

    print(f'[i] Finished fitting simple distributions')

    # here fit the best distribution with iminuit and check if how high the chi2 is
    # then it is relative to your interpretation and the domain knowledge
    # but a chi2 of 1 is a good fit, anything very high or very low is a bad fit and means a complex fit is needed

    return (total_conditions,total_responses,total_not_enough_values,zero_bins,single_bins,multiple_bins,best_dist_counts,not_enough_data_counts)

def run_for_all_logs():
    log_names = ['Traffic Fine','Municipal','Hospital','Loan']

    log_paths = ['data/Road_Traffic_Fine_Management_Process.xes', #traffic
            'data/BPIC15_1.xes', #municipality
            'data/Hospital_log.xes', #hospital
            'data/BPI_Challenge_2012.xes' #loan
            ]

    res_graphs = ['models/road_traffic_fine_model',
                  'models/municipality_model',
                  'models/hospital_model',
                  'models/loan_model'
                  ]

    res_timings = ['models/road_traffic_fine_timings/',
                    'models/municipality_timings/',
                    'models/hospital_timings/',
                    'models/loan_timings/']

    res = pd.DataFrame(columns=log_names)
    tc = {}
    tr = {}
    tnev= {}
    condition_results = {}
    cond_dist_res = {}
    response_results = {}
    resp_dist_res = {}
    for i in range(0,len(log_paths)):
        total_conditions, total_responses, total_not_enough_values, zero_bins, single_bins, multiple_bins, best_dist_counts, not_enough_data_counts = run_for_one_log(log_paths[i],res_graphs[i],res_timings[i])
        tc[log_names[i]] = total_conditions
        tr[log_names[i]] = total_responses
        tnev[log_names[i]] = total_not_enough_values
        condition_results[log_names[i]] = {
            'hist: 1 bin count': single_bins['CONDITION'],
            'hist: 0 bin count': zero_bins['CONDITION'],
            'hist: multiple bin counts': multiple_bins['CONDITION'],
            'dist: not enough data': not_enough_data_counts['CONDITION']
        }
        cond_dist_res[log_names[i]] = best_dist_counts['CONDITION']

        response_results[log_names[i]] = {
            'hist: 1 bin count': single_bins['RESPONSE'],
            'hist: 0 bin count': zero_bins['RESPONSE'],
            'hist: multiple bin counts': multiple_bins['RESPONSE'],
            'dist: not enough data': not_enough_data_counts['RESPONSE']
        }
        resp_dist_res[log_names[i]] = best_dist_counts['RESPONSE']

    res = res.append(tc,ignore_index=True)
    res = res.append(tr,ignore_index=True)
    res = res.append(tnev,ignore_index=True)
    res = res.append(condition_results,ignore_index=True)
    res = res.append(cond_dist_res,ignore_index=True)
    res = res.append(response_results,ignore_index=True)
    res = res.append(resp_dist_res,ignore_index=True)
    res.to_csv('models/results.csv')
    print('[i] Finished running on all logs!')

def get_model(log_path, graph_path, log = None):
    disc = Discover()

    if log is None:
        log = load_log(log_path)
    # Mine the graph using discover
    mine_dcr(disc,log,graph_path,findAdditionalConditions=True)
    #TODO: Get events from a BPMN model and their relations + a specification whether it's mining a delay or a deadline
    #extended bpmn diagrams

def advanced_timings_fit(log = None):  # old main
    #TODO: Get events from a user specified list of events + a specification whether it's mining a delay or a deadline
    log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
    graph_path = 'models/road_traffic_fine'
    results_folder = 'models/road_traffic_fine_timings/'

    fitting_parametric_dist = Parametric()
    disc = Discover()

    if log is None:
        log = load_log(log_path)

    timings_raw_data = mine_timings(log,graph_path) # timings is a dict with a tuple for the rule and all the deltas in days (maybe change to seconds in resolution if some other dataset is used)
    # timings into boxplots
    print(f'[i] Started creating boxplot data')
    boxplot_values = create_timing_box_plots(timings_raw_data,results_folder)
    print(f'[i] Finished creating boxplot data')
    #(lower_whisker, lower_quartile, median, upper_quartile, upper_whisker, iqr)
    print(f'[i] Started writing median values as delays and deadlines to DCR graph')
    to_print_values = {}
    to_print_values_outliers = {}
    for (k,v) in boxplot_values.items():
        if v:
            if k[0] == 'CONDITION': #min for condition
                to_print_values[k] = v[6]
                to_print_values_outliers[k] = v[0]
            elif k[0] == 'RESPONSE': #max for response
                to_print_values[k] = v[7]
                to_print_values_outliers[k] = v[4]
    disc.writeGraph(graph_path,to_print_values)
    disc.writeGraph('models/road_traffic_fine_no_outliers.txt',to_print_values_outliers)

    mean_values = get_mean_values(timings_raw_data)
    print(f'[i] Finished writing median values as delays and deadlines to DCR graph')
    # timings into histograms
    print(f'[i] Started creating histograms')
    histogram_values = create_histograms(timings_raw_data,results_folder,xmin=0,xmax=800)
    print(f'[i] Finished creating histograms')
    # histograms + overlay of simple distributions fitted
    #TODO 18.03.2022 Once a list of best fitting functions is provided fit them with iminuit and chi2 to get a nice output.
    print(f'[i] Started fitting simple distributions')
    single_distribution_fits = fitting_parametric_dist.simple_distribution_fit_all_timings(timings_raw_data,results_folder,xmax=800)
    print(f'[i] Finished fitting simple distributions')

    # all pairs of functions and initial guesses for parameters in a dict matching the keys of the timing raw data
    funcs = Functions()
    functions_to_fit = {
        ('CONDITION', 'Create Fine', 'Insert Date Appeal to Prefecture'): funcs.double_log_gaussian_exp,
        ('CONDITION', 'Create Fine', 'Send Fine'): funcs.gaus_log_gauss_exp,
        ('CONDITION', 'Create Fine', 'Add penalty'): funcs.triple_gauss_log_log,
        ('CONDITION', 'Insert Fine Notification', 'Notify Result Appeal to Offender'): funcs.N_log_gauss_pdf,
        ('CONDITION', 'Add penalty', 'Send for Credit Collection'): None,
        ('CONDITION', 'Send Fine', 'Send for Credit Collection'): None,
        ('CONDITION', 'Create Fine', 'Payment'): funcs.double_log_gaussian_exp,
        ('CONDITION', 'Create Fine', 'Insert Date Appeal to Prefecture'): funcs.triple_gauss_gauss_log,
        ('CONDITION', 'Insert Fine Notification', 'Add penalty'): 60,
        ('CONDITION', 'Send Fine', 'Receive Result Appeal from Prefecture'): funcs.N_log_gauss_pdf,
        ('CONDITION', 'Create Fine', 'Appeal to Judge'): funcs.double_log_gaussian_exp,
        ('CONDITION', 'Create Fine', 'Send Appeal to Prefecture'): funcs.double_gaussian,
        ('CONDITION', 'Send Fine', 'Insert Fine Notification'): None,
        ('CONDITION', 'Create Fine', 'Notify Result Appeal to Offender'): funcs.N_gauss_pdf,
        ('RESPONSE', 'Insert Fine Notification', 'Add penalty'): 60
    }
    initial_values_dict = {
        ('CONDITION', 'Create Fine', 'Insert Date Appeal to Prefecture'): {
            'N_exp': 10000,
            'tau': 1 / np.e,
            'N': 2000,
            'mu': np.log(50),
            'sigma': np.log(10),
            'N2': 1000,
            'mu2': np.log(140),
            'sigma2': np.log(10)
        },
        ('CONDITION', 'Create Fine', 'Send Fine'): {
            'N_exp': 100000,
            'tau': 1.0,
            'N': 2000,
            'mu': 10,
            'sigma': 4,
            'N2': 1000,
            'mu2': np.log(19),
            'sigma2': np.log(3)
        },
        ('CONDITION', 'Create Fine', 'Add penalty'): {
            'N': 100,
            'mu': 60,
            'sigma': 0.5,
            'N2': 1000,
            'mu2': np.log(120),
            'sigma2': np.log(15),
            'N3': 2000,
            'mu3': np.log(205),
            'sigma3': np.log(20)
        },
        ('CONDITION', 'Insert Fine Notification', 'Notify Result Appeal to Offender'): {
            'N': 300,
            'mu': np.log(120),
            'sigma': np.log(30)
        },
        ('CONDITION', 'Add penalty', 'Send for Credit Collection'): None,
        ('CONDITION', 'Send Fine', 'Send for Credit Collection'): None,
        ('CONDITION', 'Create Fine', 'Payment'): {
            'N_exp': 1000,
            'tau': 1 / np.e,
            'N': 100,
            'mu': np.log(90),
            'sigma': np.log(10),
            'N2': 120,
            'mu2': np.log(190),
            'sigma2': np.log(10)
        },
        ('CONDITION', 'Create Fine', 'Insert Date Appeal to Prefecture'): {
            # 'N_exp':1000,
            # 'tau': 1/np.e,
            'N': 1000,
            'mu': 90,
            'sigma': 10,
            'N2': 200,
            'mu2': 19,
            'sigma2': 4,
            'N3': 1200,
            'mu3': np.log(190),
            'sigma3': np.log(10)
        },
        ('CONDITION', 'Insert Fine Notification', 'Add penalty'): None,
        ('CONDITION', 'Send Fine', 'Receive Result Appeal from Prefecture'): {
            'N': 400,
            'mu': np.log(120),
            'sigma': np.log(40)
        },
        ('CONDITION', 'Create Fine', 'Appeal to Judge'): {
            'N_exp': 1000,
            'tau': 1 / np.e,
            'N': 100,
            'mu': np.log(150),
            'sigma': np.log(50),
            'N2': 200,
            'mu2': np.log(200),
            'sigma2': np.log(10)
        },
        ('CONDITION', 'Create Fine', 'Send Appeal to Prefecture'): {
            'N': 499,
            'mu': 120,
            'sigma': 20,
            'N2': 200,
            'mu2': 200,
            'sigma2': 10
        },
        ('CONDITION', 'Send Fine', 'Insert Fine Notification'): None,
        ('CONDITION', 'Create Fine', 'Notify Result Appeal to Offender'):
            {
                'N': 200,
                'mu': 270,
                'sigma': 60
            },
        ('RESPONSE', 'Insert Fine Notification', 'Add penalty'): None
    }
    # histograms + overlay of advanced distributions fitted
    print(f'[i] Started fitting advanced distributions')
    advanced_distribution_fits = fitting_parametric_dist.advanced_distribution_fit_all_timings(timings_raw_data,functions_to_fit,initial_values_dict,results_folder,xmin=0,xmax=800)
    print(f'[i] Finished fitting advanced distributions')

# Press the green button in the gutter to run the script.
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "haftmi", ["help","all", "fine","timing","model","interactive"])
    except getopt.GetoptError:
        print('[x] main.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print('[!] main.py [-h , -a, -f, -t, -m, -i, --help, --all, --fine, --timing, --model, --interactive]')
            sys.exit()
        elif opt in ("-i","--interactive"):
            print('[i] In interactive mode')
            command = input("[i] Type your input (f,a,m) or q to quit:")
            run = True
            unknown = False
            while run:
                if command == 'f':
                    log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
                    graph_path = 'models/road_traffic_fine'
                    log = load_log(log_path)
                    get_model(log_path, graph_path,log)
                    advanced_timings_fit(log) # advanced fit on timings
                elif command == 'a':
                    run_for_all_logs() # experiment on all logs
                elif command == 'm':
                    log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
                    graph_path = 'models/road_traffic_fine'
                    get_model(log_path, graph_path)
                elif command == 'q':
                    run = False
                else:
                    print(f'[!] Unknown command {command}')
                    unknown = True
                if not unknown:
                    print(f'[i] Command {command} executed!')

        elif opt in ("-f", "--fine"):
            log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
            graph_path = 'models/road_traffic_fine'
            log = load_log(log_path)
            get_model(log_path, graph_path,log)
            advanced_timings_fit(log) # advanced fit on timings
        elif opt in ("-m", "--model"):
            log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
            graph_path = 'models/road_traffic_fine'
            get_model(log_path, graph_path)
        elif opt in ("-a", "--all"):
            run_for_all_logs() # experiment on all logs
    print('[i] Done!')

if __name__ == '__main__':
    main(sys.argv[1:])
