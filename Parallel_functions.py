#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:47:46 2023

@author: julienballbe
"""
import sys
sys.path.insert(0, '/Users/julienballbe/My_Work/My_Librairies/Electrophy_treatment.py')
import Electrophy_treatment as ephys_treat
import Fit_library as fitlib
import Data_treatment as data_treat
import Parallel_functions as parafunc
import numpy as np
import pandas as pd
import h5py





def get_sweep_TPC_parallel(current_sweep, cell_id, database, my_Cell_data, stim_start_time=None, stim_end_time=None):
    '''
    Function called in concurrent runs from "create_cell_Full_TPC_table_concurrent_runs" function. 
    Create the TPC table for a given sweep

    Parameters
    ----------
    current_sweep : int

    cell_id : str or int

    database : str

    my_Cell_data 
        For Allen data, my_Cell_data = ctc.get_ephys_data(cell_id)
    stim_start_time : float

    stim_end_time : float


    Returns
    -------
    TPC_new_line : pd.Series
        first column ('Sweep')contains Sweep_id, and second column ('TPC') contains a TPC pd.DataFrame (Contains Time, Current,Potential, First Time deriv of Potential, and Second Time deriv of Potential)

    '''
    if database == 'Allen_Cell_Type_Database':
        #my_Cell_data = ctc.get_ephys_data(cell_id)
        index_range = my_Cell_data.get_sweep(current_sweep)["index_range"]

        sampling_rate = my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
        current_stim_array = (my_Cell_data.get_sweep(current_sweep)[
                              "stimulus"][0:index_range[1]+1]) * 1e12  # to pA
        current_membrane_trace = (my_Cell_data.get_sweep(current_sweep)[
                                  "response"][0:index_range[1]+1]) * 1e3  # to mV
        current_membrane_trace = np.array(current_membrane_trace)
        current_stim_array = np.array(current_stim_array)
        stim_start_index = index_range[0]+next(x for x, val in enumerate(
            current_stim_array[index_range[0]:]) if val != 0)
        current_time_array = np.arange(
            0, len(current_stim_array)) * (1.0 / sampling_rate)

        stim_start_time = current_time_array[stim_start_index]
        stim_end_time = stim_start_time+1.
        
        
        original_TPC = ephys_treat.create_TPC(time_trace=current_time_array,
                                              potential_trace=current_membrane_trace,
                                              current_trace=current_stim_array,
                                              filter=5.)

        original_TPC = original_TPC[original_TPC['Time_s'] < (
            stim_end_time+.5)]
        original_TPC = original_TPC[original_TPC['Time_s'] > (
            stim_start_time-.5)]
        original_TPC = original_TPC.apply(pd.to_numeric)
        original_TPC = original_TPC.reset_index(drop=True)

        original_TPC.index = original_TPC.index.astype(int)

        
        
       

    elif database == 'Lantyer_Database':

        current_time_array, current_membrane_trace, current_stim_array = data_treat.get_traces(
            cell_id, current_sweep, 'Lantyer_Database')[:3]
        original_TPC = ephys_treat.create_TPC(time_trace=current_time_array,
                                              potential_trace=current_membrane_trace,
                                              current_trace=current_stim_array,
                                              filter=5.)
        # original_TPC=original_TPC[original_TPC['Time_s']<(stim_end_time+.5)]
        # original_TPC=original_TPC[original_TPC['Time_s']>(stim_start_time-.5)]
        original_TPC = original_TPC.apply(pd.to_numeric)
        original_TPC = original_TPC.reset_index(drop=True)

        original_TPC.index = original_TPC.index.astype(int)

        
        
    elif database == 'NVC':
        current_time_array, current_membrane_trace, current_stim_array,stim_start_time, stim_end_time = data_treat.get_traces(
            cell_id, current_sweep, 'NVC')
        
        original_TPC = ephys_treat.create_TPC(time_trace=current_time_array,
                                              potential_trace=current_membrane_trace,
                                              current_trace=current_stim_array,
                                              filter=5.)

        original_TPC = original_TPC[original_TPC['Time_s'] < (
            stim_end_time+.5)]
        original_TPC = original_TPC[original_TPC['Time_s'] > (
            stim_start_time-.5)]
        original_TPC = original_TPC.apply(pd.to_numeric)
        original_TPC = original_TPC.reset_index(drop=True)

        original_TPC.index = original_TPC.index.astype(int)

        
    
    TPC_new_line = pd.DataFrame(
        [str(current_sweep), original_TPC]).T
    TPC_new_line.columns=['Sweep', 'TPC']
    return TPC_new_line



def get_TPC_SF_table_concurrent_runs_bis(file, sweep):

    current_file = h5py.File(file, 'r')
    TPC_group = current_file['TPC_tables']
    current_TPC_table = TPC_group[str(sweep)]
    
    TPC_colnames = np.array([x.decode('ascii')
                            for x in list(TPC_group['TPC_colnames'])], dtype='str')
    #TPC_colnames = list(TPC_group['TPC_colnames'])
    current_TPC_table = pd.DataFrame(current_TPC_table)
    current_TPC_table.columns = TPC_colnames
    
    new_line_TPC = pd.DataFrame(
        [str(sweep), current_TPC_table]).T
    new_line_TPC.columns=['Sweep', 'TPC']
    
    
    SF_group = current_file['SF_tables']
    current_SF_table = SF_group[str(sweep)]
    SF_dict = {}

    for feature in current_SF_table.keys():
        SF_dict[feature] = np.array(current_SF_table[feature])

    new_line_SF = pd.DataFrame([str(sweep), SF_dict]).T
    new_line_SF.columns=['Sweep', 'SF_dict']

    return new_line_TPC, new_line_SF

def get_sweep_info_parallel(cell_Full_TPC_table, current_sweep, cell_id, database, stim_start_time, stim_end_time,ctc_cell_data=None):

    if database == 'Allen_Cell_Type_Database':

        my_Cell_data = ctc_cell_data
        
        # stim_amp = my_Cell_data.get_sweep_metadata(
        #     current_sweep)['aibs_stimulus_amplitude_pa']

        Protocol_id = 1
        trace_id = current_sweep

        index_range = my_Cell_data.get_sweep(int(current_sweep))["index_range"]
        sampling_rate = my_Cell_data.get_sweep(int(current_sweep))["sampling_rate"]

        current_stim_array = (my_Cell_data.get_sweep(int(current_sweep))[
                              "stimulus"][0:index_range[1]+1]) * 1e12  # to pA
        current_stim_array = np.array(current_stim_array)

        stim_start_index = index_range[0]+next(x for x, val in enumerate(
            current_stim_array[index_range[0]:]) if val != 0)
        current_time_array = np.arange(
            0, len(current_stim_array)) * (1.0 / sampling_rate)

        stim_start_time = current_time_array[stim_start_index]
        stim_end_time = stim_start_time+1.
        
        current_TPC=data_treat.get_filtered_TPC_table(cell_Full_TPC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)

        
        stim_baseline,stim_amp = fitlib.fit_stimulus_trace(current_TPC, stim_start_time, stim_end_time)[:2]

        # Bridge_Error, membrane_time_cst, R_in, V_rest = ephys_treat.estimate_bridge_error(
        #     current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)
        
        Bridge_Error= ephys_treat.estimate_bridge_error(
            current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)

        if np.isnan(Bridge_Error):
            BE_extrapolated = True
        else:
            BE_extrapolated = False

        # output_line = pd.Series([current_sweep,
        #                          Protocol_id,
        #                          trace_id,
        #                          stim_amp,
        #                          stim_start_time,
        #                          stim_end_time,
        #                          Bridge_Error,
        #                          BE_extrapolated,
        #                          membrane_time_cst,
        #                          R_in,
        #                          V_rest,
        #                          sampling_rate], index=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', "Time_constant_ms", 'Input_Resistance_MOhm', 'Membrane_resting_potential_mV', 'Sampling_Rate_Hz'])

        
    elif database == 'Lantyer_Database':
        cell_sweep_stim_table = pd.read_pickle(
            '/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table = cell_sweep_stim_table[cell_sweep_stim_table['Cell_id'] == cell_id]
        
        
        current_TPC=data_treat.get_filtered_TPC_table(cell_Full_TPC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)
        
        time_trace = np.array(current_TPC['Time_s'])
        experiment,Protocol_id,trace_id = current_sweep.split("_")
        # Protocol_id = int(str(current_sweep)[0])
        # trace_id = int(str(current_sweep)[1:])

        sampling_rate = np.rint(1/(time_trace[1]-time_trace[0]))

        # stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')[-2:]

        # stim_start_time,stim_end_time=fitlib.fit_stimulus_trace(current_TPC,np.nan,np.nan,do_plot=False)[2:4]
        stim_baseline,stim_amp = fitlib.fit_stimulus_trace(
            current_TPC, stim_start_time, stim_end_time)[:2]
        # Bridge_Error, membrane_time_cst, R_in, V_rest = ephys_treat.estimate_bridge_error(
        #     current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)
        
        Bridge_Error= ephys_treat.estimate_bridge_error(
            current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)
        
        if np.isnan(Bridge_Error):
            BE_extrapolated = True
        else:
            BE_extrapolated = False

        # output_line = pd.Series([current_sweep,
        #                          Protocol_id,
        #                          trace_id,
        #                          stim_amp,
        #                          stim_start_time,
        #                          stim_end_time,
        #                          Bridge_Error,
        #                          BE_extrapolated,
        #                          membrane_time_cst,
        #                          R_in,
        #                          V_rest,
        #                          sampling_rate], index=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', "Time_constant_ms", 'Input_Resistance_MOhm', 'Membrane_resting_potential_mV', 'Sampling_Rate_Hz'])

        
    elif database == 'NVC':
        
        cell_file_directory = pd.read_csv('/Volumes/Work_Julien/NVC_database/Cell_files_id_directories.csv',index_col=None)
        current_table_CC = cell_file_directory[cell_file_directory['Cell_id']==cell_id ]
        current_table_CC = current_table_CC[current_table_CC['CC/DC']=='CC']
        current_table_CC = current_table_CC.sort_values(
            by=['Time'])
        current_table_CC=current_table_CC.reset_index(drop=True)
        
        experiment,Protocol_id,trace_id = current_sweep.split("_")
        Protocol_id = int(Protocol_id)
        trace_id = int(trace_id)
        # current_experiment = int(str(current_sweep)[0])
        # Protocol_id = int(str(current_sweep)[1:3])
        # trace_id = int(str(current_sweep)[3:])
        
        current_time_file=current_table_CC.loc[Protocol_id-1,'Time']
        
        file_path=data_treat.build_file_path_from_cell_id_NVC(cell_id,cell_file_directory)
        Full_cell_file='/Volumes/Work_Julien/NVC_database/Original_Data/'+file_path+'/G clamp/'+current_time_file
        

        float_array=np.fromfile(Full_cell_file,dtype='>f')
        separator_indices=np.argwhere(float_array==max(float_array))   
        header=float_array[:separator_indices[0][0]]

        metadata={"Iterations":header[0],
                  "I_start_pulse_pA":header[1],
                  "I_increment_pA":header[2],
                  "I_pulse_width_ms":header[3],
                  "I_delay_ms":header[4],
                  "Sweep_duration_ms":header[5],
                  "Sample_rate_Hz":header[6],
                  "Cycle_time_ms":header[7],
                  "Bridge_MOhms":header[8],
                  "Cap_compensation":header[9]}
        
        stim_start_time=metadata["I_delay_ms"]*1e-3
        stim_end_time=stim_start_time+metadata["I_pulse_width_ms"]*1e-3
        sampling_rate=metadata['Sample_rate_Hz']
        
        current_TPC=data_treat.get_filtered_TPC_table(cell_Full_TPC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)
        
        stim_baseline,stim_amp = fitlib.fit_stimulus_trace(
            current_TPC, stim_start_time, stim_end_time)[:2]
        
        # if np.isnan(metadata['I_start_pulse_pA']) or np.isnan(metadata['I_increment_pA']):
        #     stim_amp=fitlib.fit_stimulus_trace(
        #         current_TPC, stim_start_time, stim_end_time)[1]
        
        # else:
        #     stim_amp = metadata['I_start_pulse_pA']+(trace_id-1)*metadata['I_increment_pA']
        
        
        
        # Bridge_Error, membrane_time_cst, R_in, V_rest = ephys_treat.estimate_bridge_error(
        #     current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)
        
        Bridge_Error= ephys_treat.estimate_bridge_error(
            current_TPC, stim_amp, stim_start_time, stim_end_time, do_plot=False)
        
        if np.isnan(Bridge_Error):
            BE_extrapolated = True
        else:
            BE_extrapolated = False
        
        # output_line = pd.Series([current_sweep,
        #                          Protocol_id,
        #                          trace_id,
        #                          stim_amp,
        #                          stim_start_time,
        #                          stim_end_time,
        #                          Bridge_Error,
        #                          BE_extrapolated,
        #                          membrane_time_cst,
        #                          R_in,
        #                          V_rest,
        #                          sampling_rate], index=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', "Time_constant_ms", 'Input_Resistance_MOhm', 'Membrane_resting_potential_mV', 'Sampling_Rate_Hz'])

        
    output_line = pd.DataFrame([str(current_sweep),
                             Protocol_id,
                             trace_id,
                             stim_amp,
                             stim_baseline,
                             stim_start_time,
                             stim_end_time,
                             Bridge_Error,
                             BE_extrapolated,
                             sampling_rate]).T
    output_line.columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_amp_pA', 'Stim_baseline_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz']

    return output_line


def estimate_sweep_stim_time_limit_parallel(cell_id, sweep, database):
    '''
    Estimate for a given sweep the start time and end time of the stimulus. 
    Assumes a 500ms duration, (for Lantyer) if otherwise, this function should be addapted with a dedicated variable

    Parameters
    ----------
    cell_id : TYPE
        DESCRIPTION.
    sweep : TYPE
        DESCRIPTION.
    database : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    current_time_array, current_membrane_trace, current_stim_array, stim_start_time, stim_end_time = data_treat.get_traces(
        cell_id, sweep, database)
    TPC = ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=5.)

    stim_table = TPC.loc[:, ['Time_s', 'Input_current_pA']]

    stim_table['Input_current_pA']=np.array(data_treat.filter_trace(stim_table['Input_current_pA'],stim_table['Time_s'],filter=5.,do_plot=False))  

    current_derivative = data_treat.get_derivative(np.array(stim_table['Input_current_pA']),
                                        np.array(stim_table['Time_s']))

    stim_table["Input_current_pA"]=np.array(data_treat.filter_trace(stim_table['Input_current_pA'],stim_table['Time_s'],filter=5.,do_plot=False)) 

    current_derivative = data_treat.get_derivative(np.array(stim_table['Input_current_pA']),
                                            np.array(stim_table['Time_s']))
    current_derivative=np.insert(current_derivative,0,np.nan)
    stim_table["Filtered_Stimulus_trace_derivative_pA/ms"]=np.array(current_derivative)
    # remove last 50ms of signal (potential step)
    limit = stim_table.shape[0] - \
        int(0.05/(stim_table.iloc[1, 0]-stim_table.iloc[0, 0]))

    stim_table.loc[limit:, "Filtered_Stimulus_trace_derivative_pA/ms"] = np.nan
    
    stim_table = data_treat.get_autocorrelation(stim_table, .500, do_plot=False)
    # best_time_autocorr=.500
    #best_stim_start_index=next(x for x, val in enumerate(stim_table['Autocorrelation']) if np.abs(val) >= 10000 )
    # best_stim_start=stim_table.iloc[best_stim_start_index,0]
    
    best_stim_start = stim_table[stim_table['Autocorrelation'] == np.nanmin(
        stim_table['Autocorrelation'])].iloc[0, 0]

    stim_end = best_stim_start+.500
    new_line = pd.Series([sweep, best_stim_start, stim_end], index=[
                         'Sweep', 'Stim_start_s', 'Stim_end_s'])
    return(new_line)
