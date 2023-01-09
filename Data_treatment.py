#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:51:51 2022

@author: julienballbe
"""

import allensdk
import json
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.internal.model.biophysical import ephys_utils
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
import time
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle,geom_vline
import scipy
from scipy.stats import linregress
from scipy import optimize
from scipy.optimize import curve_fit
import random
from tqdm import tqdm

import logging
from allensdk.core.swc import Marker
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import warnings
from allensdk.core.cell_types_cache import CellTypesCache
import pandas
from allensdk.api.queries.cell_types_api import CellTypesApi
import os
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel
from lmfit import Parameters, Minimizer,fit_report
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
from plotnine.labels import xlab
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sys
sys.path.insert(0, '/Users/julienballbe/My_Work/My_Librairies/Electrophy_treatment.py')
import Electrophy_treatment as ephys_treat
import Fit_library as fitlib
import functools
import h5py
import glob
from datetime import date
import concurrent.futures

#%%
ctc= CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
#%%


def data_pruning (stim_freq_table):
    
    stim_freq_table=stim_freq_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    frequency_array=np.array(stim_freq_table.loc[:,'Frequency_Hz'])
    step_array=np.diff(frequency_array)
    non_zero_freq=frequency_array[np.where(frequency_array>0)]
    
    if np.count_nonzero(frequency_array)<4:
        obs='Less_than_4_response'
        do_fit=False
        return obs,do_fit
    
    if len(np.unique(non_zero_freq))<3 :
        obs='Less_than_3_different_frequencies'
        do_fit=False
        return obs,do_fit
        
    first_non_zero=np.flatnonzero(frequency_array)[0]
    stim_array=np.array(stim_freq_table.loc[:,'Stim_amp_pA'])[(first_non_zero-1):]
    stimulus_span=stim_array[-1]-stim_array[0]
    
    # if stimulus_span<100.:
    #     obs='Stimulus_span_lower_than_100pA'
    #     do_fit=False
    #     return obs,do_fit
    
    # count,bins=np.histogram(stim_array,
    #      bins=int((stim_array[-1]+(10 - stimulus_span % 10)-stim_array[0])/10),
    #       range=(stim_array[0], stim_array[-1]+(10 - stimulus_span % 10)))
          
        
    # different_stim=len(np.flatnonzero(count))
    
    # if different_stim <5:
    #     obs='Less_than_5_different_stim_amp'
    #     do_fit=False
    #     return obs,do_fit
    
    obs='-'
    do_fit=True
    return obs,do_fit


def data_pruning_adaptation (original_SF_table,original_cell_sweep_info_table,response_time):
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    SF_table=original_SF_table.copy()
    maximum_nb_interval =0
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])
        
        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes
    if maximum_nb_interval<4:
        do_fit=False
        adapt_obs='Less_than_4_intervals_at_most'
    else:
        do_fit=True
        adapt_obs='-'
    return adapt_obs,do_fit


    
def import_spike_time_table(cell_id,original_database):
    if original_database=='Lantyer':
        stim_spike_file= pd.read_csv(filepath_or_buffer=str('/Users/julienballbe/My_Work/Lantyer_Data/Stim_spike_tables/'+str(cell_id)+'_stim_spike_table.csv'))
        stim_spike_file=stim_spike_file.loc[:,'Trace_id':]
        stim_spike_table=stim_spike_table=pd.DataFrame(columns=['Trace_id',
              "Sweep_id",
              'Stim_amp_pA',
              'Stim_baseline_pA',
              'Stim_start_s',
              'Stim_end_s', 
              'Membrane_potential_SS_mV',
              "Membrane_baseline_mV",
              'Spike_times_s',
              'Spike_thresh_time_s',
              'Spike_thresh_pot_mV',
              'Spike_peak_time_s',
              'Spike_peak_pot_mV',
              'Spike_upstroke_time_s',
              'Spike_upstroke_pot_mV',
              'Spike_downstroke_time_s',
              'Spike_downstroke_pot_mV',
              'Trough_time_s',
              'Trough_pot_mV'])
        
        for line in range(stim_spike_file.shape[0]):
            new_line=pd.Series([str(stim_spike_file.loc[line,"Trace_id"]),
                          int(stim_spike_file.loc[line,"Sweep_id"]),
                          stim_spike_file.loc[line,"Stim_amp_pA"],
                          stim_spike_file.loc[line,"Stim_baseline_pA"],
                          stim_spike_file.loc[line,"Stim_start_s"],
                          stim_spike_file.loc[line,"Stim_end_s"],
                          stim_spike_file.loc[line,"Membrane_potential_SS_mV"],
                          stim_spike_file.loc[line,"Membrane_baseline_mV"],
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_times_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_thresh_time_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_thresh_pot_mV"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_peak_time_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_peak_pot_mV"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_upstroke_time_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_upstroke_pot_mV"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_downstroke_time_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_downstroke_pot_mV"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Trough_time_s"])),
                          np.array(pd.eval(stim_spike_file.loc[line,"Trough_pot_mV"]))],
                                index=['Trace_id',
                                      "Sweep_id",
                                      'Stim_amp_pA',
                                      'Stim_baseline_pA',
                                      'Stim_start_s',
                                      'Stim_end_s', 
                                      'Membrane_potential_SS_mV',
                                      "Membrane_baseline_mV",
                                      'Spike_times_s',
                                      'Spike_thresh_time_s',
                                      'Spike_thresh_pot_mV',
                                      'Spike_peak_time_s',
                                      'Spike_peak_pot_mV',
                                      'Spike_upstroke_time_s',
                                      'Spike_upstroke_pot_mV',
                                      'Spike_downstroke_time_s',
                                      'Spike_downstroke_pot_mV',
                                      'Trough_time_s',
                                      'Trough_pot_mV'])
            stim_spike_table = stim_spike_table.append(new_line, ignore_index=True)
    
    elif original_database=='Allen':
        stim_spike_file= pd.read_csv(filepath_or_buffer=str('/Users/julienballbe/My_Work/Allen_Data/Stim_spike_tables/'+str(cell_id)+'_stim_spike_table.csv'))
        stim_spike_file=stim_spike_file.loc[:,'Cell_id':]
        stim_spike_table=stim_spike_table=pd.DataFrame(columns=['Cell_id',
              "Sweep_id",
              'Stim_amp_pA',
              'Stim_start_s',
              'Stim_end_s', 
              'Spike_times_s'])
        
        for line in range(stim_spike_file.shape[0]):
            new_line=pd.Series([str(stim_spike_file.loc[line,"Cell_id"]),
                          int(stim_spike_file.loc[line,"Sweep_id"]),
                          stim_spike_file.loc[line,"Stim_amp_pA"],

                          stim_spike_file.loc[line,"Stim_start_s"],
                          stim_spike_file.loc[line,"Stim_end_s"],
                          np.array(pd.eval(stim_spike_file.loc[line,"Spike_times_s"]))],
                                index=['Cell_id',
                                      "Sweep_id",
                                      'Stim_amp_pA',
                                      'Stim_start_s',
                                      'Stim_end_s',
                                      'Spike_times_s'])
            stim_spike_table = stim_spike_table.append(new_line, ignore_index=True)
    return stim_spike_table
#to get spike time : x=pd.eval(mytest.iloc[7,6])


def create_cell_feature_table(cell_id_list,table_5,table_10,table_25,table_50,table_100,table_250,table_500):
    file_list=[table_10,table_25,table_50,table_100,table_250,table_500]
    time_list=['10','25','50',"100","250","500"]
    for cell in cell_id_list:
        sub_table=pd.DataFrame(columns=['Cell_id','Response_time_ms','Fit','Gain','Threshold','Saturation'])
        new_line=pd.Series(np.insert(np.array(table_5[table_5['Cell_id']==str(cell)]),1,'5'),index=['Cell_id','Response_time_ms','Fit','Gain','Threshold','Saturation'])
        sub_table=sub_table.append(new_line,ignore_index=True)

        for file in range(len(file_list)):
            new_line=pd.Series(np.insert(np.array(file_list[file][file_list[file]['Cell_id']==str(cell)]),1,time_list[file]),index=['Cell_id','Response_time_ms','Fit','Gain','Threshold','Saturation'])
            sub_table=sub_table.append(new_line,ignore_index=True)
        sub_table.to_csv('/Users/julienballbe/My_Work/Allen_Data/Cell_Features_table/'+str(cell)+'_Features_table.csv')
    
    
def create_cell_fit_table(cell_id_list,table_5,table_10,table_25,table_50,table_100,table_250,table_500):
    file_list=[table_10,table_25,table_50,table_100,table_250,table_500]
    time_list=['10','25','50',"100","250","500"]
    for cell in cell_id_list:
        sub_table=pd.DataFrame(columns=['Cell_id','Response_time_ms','Fit','QNRMSE','Amplitude','Center','Sigma', 'best_compo_QNRMSE',
        'best_heaviside_step', 'best_sigmoid_amplitude', 'best_sigmoid_center','best_sigmoid_sigma'])
        new_line=pd.Series(np.insert(np.array(table_5[table_5['Cell_id']==str(cell)]),1,'5'),index=['Cell_id','Response_time_ms','Fit','QNRMSE','Amplitude','Center','Sigma', 'best_compo_QNRMSE',
        'best_heaviside_step', 'best_sigmoid_amplitude', 'best_sigmoid_center','best_sigmoid_sigma'])

        sub_table=sub_table.append(new_line,ignore_index=True)

        for file in range(len(file_list)):
            new_line=pd.Series(np.insert(np.array(file_list[file][file_list[file]['Cell_id']==str(cell)]),1,time_list[file]),index=['Cell_id','Response_time_ms','Fit','QNRMSE','Amplitude','Center','Sigma', 'best_compo_QNRMSE',
            'best_heaviside_step', 'best_sigmoid_amplitude', 'best_sigmoid_center','best_sigmoid_sigma'])
            sub_table=sub_table.append(new_line,ignore_index=True)
        sub_table=sub_table.loc[:,:'Sigma']
        sub_table.to_csv('/Users/julienballbe/My_Work/Allen_Data/Cell_Fit_table/'+str(cell)+'_Fit_table.csv') 
        
def coarse_or_fine_sweep(cell_id,stimulus_type="Long Square"):
    all_sweeps_table=pd.DataFrame(ctc.get_ephys_sweeps(cell_id))
    all_sweeps_table=all_sweeps_table[all_sweeps_table['stimulus_name']==stimulus_type]
    all_sweeps_table=all_sweeps_table[['sweep_number','stimulus_description']]
    for elt in range(all_sweeps_table.shape[0]):
        if 'COARSE' in all_sweeps_table.iloc[elt,1]:
            all_sweeps_table.iloc[elt,1]='COARSE'
        elif 'FINEST' in all_sweeps_table.iloc[elt,1]:
            all_sweeps_table.iloc[elt,1]='FINEST'    
    return(all_sweeps_table)

def create_raw_trace_file(cell_id):
    my_Cell_data = ctc.get_ephys_data(cell_id)
    sweep_type=coarse_or_fine_sweep(cell_id)
    sweep_type=sweep_type[sweep_type["stimulus_description"]=="COARSE"]
    coarse_sweep_number=sweep_type.loc[:,"sweep_number"].values
    raw_dataframe=pd.DataFrame(columns=['Sweep','Stim_start_s','Stim_end_s','Time_array','Potential_array','Current_array'])
    
    for current_sweep in coarse_sweep_number:
        
        
        index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
        
        sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
        current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
        current_membrane_trace=(my_Cell_data.get_sweep(current_sweep)["response"][0:index_range[1]+1])* 1e3 # to mV
        current_membrane_trace=np.array(current_membrane_trace)
        current_stim_array=np.array(current_stim_array)
        stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
        current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
        
        
        
        stim_start_time=current_time_array[stim_start_index]
        stim_end_time=stim_start_time+1.
        
        cutoff_start_idx=find_time_index(current_time_array,(stim_start_time-.1))
        cutoff_end_idx=find_time_index(current_time_array,(stim_end_time+.1))
        
        current_stim_array=current_stim_array[cutoff_start_idx:cutoff_end_idx]
        current_membrane_trace=current_membrane_trace[cutoff_start_idx:cutoff_end_idx]
        
        new_line=pd.Series([current_sweep,stim_start_time,stim_end_time,sampling_rate,current_membrane_trace,current_stim_array],
                           index=['Sweep','Stim_start_s','Stim_end_s','Time_array','Potential_array','Current_array'])
        
        raw_dataframe=raw_dataframe.append(new_line,ignore_index=True)
    raw_dataframe.to_pickle('/Volumes/easystore/Raw_traces/'+str(cell_id)+'_raw_traces.pkl')
     
        
def find_time_index(t, t_0):
    """ Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """
    assert t[0] <= t_0 <= t[-1], "Given time ({:f}) is outside of time range ({:f}, {:f})".format(t_0, t[0], t[-1])

    idx = np.argmin(abs(t - t_0))
    return idx

def create_cell_full_TPC_table (cell_id, database):
    full_TPC_table=pd.DataFrame(columns=['Sweep','TPC'])
    if database=='Allen':
        ctc=CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        my_Cell_data = ctc.get_ephys_data(cell_id)
        sweep_type=coarse_or_fine_sweep(cell_id)
        sweep_type=sweep_type[sweep_type["stimulus_description"]=="COARSE"]
        coarse_sweep_number=sweep_type.loc[:,"sweep_number"].values
        for current_sweep in coarse_sweep_number:
            
            index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
            
            sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
            current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
            current_membrane_trace=(my_Cell_data.get_sweep(current_sweep)["response"][0:index_range[1]+1])* 1e3 # to mV
            current_membrane_trace=np.array(current_membrane_trace)
            current_stim_array=np.array(current_stim_array)
            stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
            current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
            
            
            stim_start_time=current_time_array[stim_start_index]
            stim_end_time=stim_start_time+1.
            original_TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            
            original_TPC=original_TPC[original_TPC['Time_s']<(stim_end_time+.5)]
            original_TPC=original_TPC[original_TPC['Time_s']>(stim_start_time-.5)]
            
            original_TPC=original_TPC.reset_index(drop=True)

            original_TPC.index=original_TPC.index.astype(int)
            
            TPC_new_line=pd.Series([current_sweep,original_TPC],index=['Sweep','TPC'])
            full_TPC_table=full_TPC_table.append(TPC_new_line,ignore_index=True)
        
        
        
    elif database== 'Lantyer':
        cell_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=cell_sweep_stim_table[cell_sweep_stim_table['Cell_id']==cell_id]
        sweep_list=np.array(cell_sweep_stim_table.iloc[0,-1])
        for current_sweep in sweep_list:
            
            current_time_array,current_membrane_trace,current_stim_array,stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')
            original_TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            original_TPC=original_TPC[original_TPC['Time_s']<(stim_end_time+.5)]
            original_TPC=original_TPC[original_TPC['Time_s']>(stim_start_time-.5)]
            original_TPC=original_TPC.reset_index(drop=True)

            original_TPC.index=original_TPC.index.astype(int)
            
            TPC_new_line=pd.Series([current_sweep,original_TPC],index=['Sweep','TPC'])
            full_TPC_table=full_TPC_table.append(TPC_new_line,ignore_index=True)    
        
    full_TPC_table.index=full_TPC_table.loc[:,'Sweep']
    
    return full_TPC_table


def create_cell_full_TPC_table_concurrent_runs (cell_id, database):
    full_TPC_table=pd.DataFrame(columns=['Sweep','TPC'])
    if database=='Allen':
        ctc=CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        my_Cell_data = ctc.get_ephys_data(cell_id)
        sweep_type=coarse_or_fine_sweep(cell_id)
        sweep_type=sweep_type[sweep_type["stimulus_description"]=="COARSE"]
        coarse_sweep_number=sweep_type.loc[:,"sweep_number"].values
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            TPC_lines_dict={executor.submit(get_sweep_TPC,x,cell_id, database,my_Cell_data): x for x in coarse_sweep_number}
            for f in concurrent.futures.as_completed(TPC_lines_dict):
                full_TPC_table=full_TPC_table.append(f.result(),ignore_index=True)
                
                
    elif database== 'Lantyer':
        cell_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=cell_sweep_stim_table[cell_sweep_stim_table['Cell_id']==cell_id]
        sweep_list=np.array(cell_sweep_stim_table.iloc[0,-1])
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            TPC_lines_dict={executor.submit(get_sweep_TPC,x,cell_id, database,'--'): x for x in sweep_list}
            for f in concurrent.futures.as_completed(TPC_lines_dict):
                full_TPC_table=full_TPC_table.append(f.result(),ignore_index=True)
        
        
        
    full_TPC_table.index=full_TPC_table.loc[:,'Sweep']
    
    return full_TPC_table

def get_sweep_TPC(current_sweep,cell_id, database,my_Cell_data):
    if database=='Allen': 
        my_Cell_data = ctc.get_ephys_data(cell_id)
        index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
        
        sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
        current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
        current_membrane_trace=(my_Cell_data.get_sweep(current_sweep)["response"][0:index_range[1]+1])* 1e3 # to mV
        current_membrane_trace=np.array(current_membrane_trace)
        current_stim_array=np.array(current_stim_array)
        stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
        current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
        
        
        stim_start_time=current_time_array[stim_start_index]
        stim_end_time=stim_start_time+1.
        original_TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                             potential_trace=current_membrane_trace,
                             current_trace=current_stim_array,
                             filter=2.)
        
        original_TPC=original_TPC[original_TPC['Time_s']<(stim_end_time+.5)]
        original_TPC=original_TPC[original_TPC['Time_s']>(stim_start_time-.5)]
        original_TPC=original_TPC.apply(pd.to_numeric)
        original_TPC=original_TPC.reset_index(drop=True)

        original_TPC.index=original_TPC.index.astype(int)
        
        TPC_new_line=pd.Series([current_sweep,original_TPC],index=['Sweep','TPC'])
    
    elif database== 'Lantyer':
        
            
        current_time_array,current_membrane_trace,current_stim_array,stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')
        original_TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                             potential_trace=current_membrane_trace,
                             current_trace=current_stim_array,
                             filter=2.)
        original_TPC=original_TPC[original_TPC['Time_s']<(stim_end_time+.5)]
        original_TPC=original_TPC[original_TPC['Time_s']>(stim_start_time-.5)]
        original_TPC=original_TPC.apply(pd.to_numeric)
        original_TPC=original_TPC.reset_index(drop=True)

        original_TPC.index=original_TPC.index.astype(int)
        
        TPC_new_line=pd.Series([current_sweep,original_TPC],index=['Sweep','TPC'])
            
    
    
    return TPC_new_line
    
    
        
def create_cell_full_SF_dict_table(original_full_TPC_table,original_cell_sweep_info_table):
    full_TPC_table=original_full_TPC_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(full_TPC_table['Sweep'])
    full_SF_dict_table=pd.DataFrame(columns=['Sweep',"SF_dict"])
    
    for current_sweep in sweep_list:
        current_TPC=full_TPC_table.loc[current_sweep,"TPC"].copy()
        membrane_trace=np.array(current_TPC['Membrane_potential_mV'])
        time_trace=np.array(current_TPC['Time_s'])
        current_trace=np.array(current_TPC['Input_current_pA'])
        stim_start=cell_sweep_info_table.loc[current_sweep,'Stim_start_s']
        stim_end=cell_sweep_info_table.loc[current_sweep,'Stim_end_s']
        
        
        SF_dict=ephys_treat.identify_spike(membrane_trace,time_trace,current_trace,stim_start,stim_end,do_plot=False)
        new_line=pd.Series([current_sweep,SF_dict],index=['Sweep','SF_dict'])
        full_SF_dict_table=full_SF_dict_table.append(new_line,ignore_index=True)
    full_SF_dict_table.index=full_SF_dict_table.loc[:,'Sweep']
    full_SF_dict_table.index=full_SF_dict_table.index.astype(int)
    
    return full_SF_dict_table

def create_cell_sweep_info_table(cell_id,database,cell_full_TPC_table):
    cell_sweep_info_table=pd.DataFrame(columns=['Sweep','Stim_amp_pA','Stim_start_s','Stim_end_s','Bridge_Error_GOhms',"Time_constant_s",'Sampling_Rate_Hz'])
    
    if database=='Allen':
        sweep_list=np.array(cell_full_TPC_table['Sweep'])
        Train_df=pd.DataFrame(columns=['Sweep','Train_id'])
        Trace_df=pd.DataFrame(columns=['Sweep','Trace_id'])
        Stim_amp_df=pd.DataFrame(columns=['Sweep','Stim_amp_pA'])
        Stim_start_df=pd.DataFrame(columns=['Sweep','Stim_start_s'])
        Stim_end_df=pd.DataFrame(columns=['Sweep','Stim_end_s'])
        Bridge_Error_df=pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms'])
        Bridge_Error_extrapolated_df=pd.DataFrame(columns=['Sweep','Bridge_Error_extrapolated'])
        Time_constant_df=pd.DataFrame(columns=['Sweep','Time_constant_ms'])
        Input_Resistance_df=pd.DataFrame(columns=['Sweep','Input_Resistance_MOhm'])
        Sampling_rate_df=pd.DataFrame(columns=['Sweep','Sampling_Rate_Hz'])
        
        my_Cell_data = ctc.get_ephys_data(cell_id)
        #Get Stim_amp for all Sweep
        for current_sweep in sweep_list:
            stim_amp=my_Cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
            stim_amp_line=pd.Series([current_sweep,stim_amp],index=['Sweep','Stim_amp_pA'])
            
            train_line=pd.Series([current_sweep,1],index=['Sweep','Train_id'])
            trace_line=pd.Series([current_sweep,current_sweep],index=['Sweep','Trace_id'])
            
            index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
            sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
            sampling_rate_line=pd.Series([current_sweep,sampling_rate],index=['Sweep','Sampling_Rate_Hz'])
            
            
            current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
            current_stim_array=np.array(current_stim_array)
            
            stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
            current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
            
            stim_start_time=current_time_array[stim_start_index]
            stim_end_time=stim_start_time+1.
            
            Stim_start_line=pd.Series([current_sweep,stim_start_time],index=['Sweep','Stim_start_s'])
            Stim_end_line=pd.Series([current_sweep,stim_end_time],index=['Sweep','Stim_end_s'])
            
            current_TPC=cell_full_TPC_table.loc[current_sweep,'TPC'].copy()

            Bridge_Error=ephys_treat.estimate_bridge_error(current_TPC, stim_amp, stim_start_time, stim_end_time,do_plot=False)

            Bridge_Error_line=pd.Series([current_sweep,Bridge_Error],index=['Sweep','Bridge_Error_GOhms'])
            if np.isnan(Bridge_Error):
                BE_extrapolated=True
            else:
                BE_extrapolated=False
            BE_extrapolated_line=pd.Series([current_sweep,BE_extrapolated],index=['Sweep','Bridge_Error_extrapolated'])
            
            time_trace=np.array(current_TPC['Time_s'])
            potential_trace=np.array(current_TPC['Membrane_potential_mV'])
            stim_input_trace=np.array(current_TPC['Input_current_pA'])
            
            start_index=current_TPC.index[current_TPC['Time_s']<=stim_start_time].to_list()

            start_index=start_index[-1]
            membrane_baseline=np.mean(potential_trace[:start_index])
            stim_baseline=np.mean(stim_input_trace[:start_index])
            
            best_A,best_tau,membrane_SS=fitlib.fit_membrane_trace(current_TPC,stim_start_time,stim_end_time,do_plot=False)
            
            Input_resistance=np.mean((membrane_SS-membrane_baseline)/(stim_amp-stim_baseline))*1e3 #Convert to MOhms
            IR_line=pd.Series([current_sweep,Input_resistance],index=['Sweep','Input_Resistance_MOhm'])
            

            time_cst=best_tau*1e3 # convert s to ms
            Time_cst_line=pd.Series([current_sweep,time_cst],index=['Sweep','Time_constant_ms'])
            
            Train_df=Train_df.append(train_line,ignore_index=True)
            Trace_df=Trace_df.append(trace_line,ignore_index=True)
            Stim_amp_df=Stim_amp_df.append(stim_amp_line,ignore_index=True)
            Stim_start_df=Stim_start_df.append(Stim_start_line,ignore_index=True)
            Stim_end_df=Stim_end_df.append(Stim_end_line,ignore_index=True)
            Bridge_Error_df=Bridge_Error_df.append(Bridge_Error_line,ignore_index=True)
            Bridge_Error_extrapolated_df=Bridge_Error_extrapolated_df.append(BE_extrapolated_line,ignore_index=True)
            Time_constant_df=Time_constant_df.append(Time_cst_line,ignore_index=True)
            Input_Resistance_df=Input_Resistance_df.append(IR_line,ignore_index=True)
            Sampling_rate_df=Sampling_rate_df.append(sampling_rate_line,ignore_index=True)
            
            
        
        
        cell_sweep_info_table= functools.reduce(lambda left, right:     # Merge all pandas DataFrames
                     pd.merge(left , right,
                              on = ["Sweep"]),
                     [Stim_amp_df,Train_df,Trace_df,Stim_start_df,Stim_end_df,Bridge_Error_df,Bridge_Error_extrapolated_df,Time_constant_df,Input_Resistance_df,Sampling_rate_df])
        cell_sweep_info_table=cell_sweep_info_table.sort_values(by=['Train_id','Trace_id','Stim_amp_pA'])
        
        extrapolated_BE = pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms'])

        for current_train in cell_sweep_info_table['Train_id'].unique():
            reduced_cell_sweep_info_table=cell_sweep_info_table[cell_sweep_info_table['Train_id']==current_train]
            BE_array=np.array(reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
            sweep_array=np.array(reduced_cell_sweep_info_table['Sweep'])
            
            
            
            nan_ind_BE = np.isnan(BE_array) 
    
            x = np.arange(len(BE_array))
            if False in nan_ind_BE:
            
                BE_array[nan_ind_BE] = np.interp(x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])
            
            
                
            extrapolated_BE_Series=pd.Series(BE_array)
            sweep_array=pd.Series(sweep_array)
            extrapolated_BE_Series=pd.DataFrame(pd.concat([sweep_array,extrapolated_BE_Series],axis=1))
            extrapolated_BE_Series.columns=['Sweep','Bridge_Error_GOhms']
            extrapolated_BE=extrapolated_BE.append(extrapolated_BE_Series,ignore_index=True)
        
        cell_sweep_info_table.pop('Bridge_Error_GOhms')

        cell_sweep_info_table=cell_sweep_info_table.merge(extrapolated_BE,how='inner', on='Sweep')

        cell_sweep_info_table=cell_sweep_info_table.loc[:,['Sweep',
                                                           "Train_id",
                                                           'Trace_id',
                                                             'Stim_amp_pA',
                                                             'Stim_start_s',
                                                             'Stim_end_s',
                                                             'Bridge_Error_GOhms',
                                                             'Bridge_Error_extrapolated',
                                                             'Time_constant_ms',
                                                             'Input_Resistance_MOhm',
                                                             'Sampling_Rate_Hz']]
        
        
        
            
        
    elif database=='Lantyer':
        sweep_list=np.array(cell_full_TPC_table['Sweep'])
        Train_df=pd.DataFrame(columns=['Sweep','Train_id'])
        Trace_df=pd.DataFrame(columns=['Sweep','Trace_id'])
        Stim_amp_df=pd.DataFrame(columns=['Sweep','Stim_amp_pA'])
        Stim_start_df=pd.DataFrame(columns=['Sweep','Stim_start_s'])
        Stim_end_df=pd.DataFrame(columns=['Sweep','Stim_end_s'])
        Bridge_Error_df=pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms'])
        Bridge_Error_extrapolated_df=pd.DataFrame(columns=['Sweep','Bridge_Error_extrapolated'])
        Time_constant_df=pd.DataFrame(columns=['Sweep','Time_constant_ms'])
        Input_Resistance_df=pd.DataFrame(columns=['Sweep','Input_Resistance_MOhm'])
        Sampling_rate_df=pd.DataFrame(columns=['Sweep','Sampling_Rate_Hz'])      
        cell_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=cell_sweep_stim_table[cell_sweep_stim_table['Cell_id']==cell_id]
        sweep_stim_table=pd.DataFrame(cell_sweep_stim_table.iloc[0,2])
        
        for current_sweep in sweep_list:
            
            current_TPC=cell_full_TPC_table.loc[current_sweep,'TPC'].copy()
            time_trace=np.array(current_TPC['Time_s'])
            potential_trace=np.array(current_TPC['Membrane_potential_mV'])
            stim_input_trace=np.array(current_TPC['Input_current_pA'])
            
            train=int(str(current_sweep)[0])
            trace=int(str(current_sweep)[1:])
            train_line=pd.Series([current_sweep,train],index=['Sweep','Train_id'])
            trace_line=pd.Series([current_sweep,trace],index=['Sweep','Trace_id'])
            
            
            stim_amp=np.float64(sweep_stim_table[sweep_stim_table['Sweep_id']==current_sweep]["Stim_amp_pA"].values)
            stim_amp_line=pd.Series([current_sweep,stim_amp],index=['Sweep','Stim_amp_pA'])
            
            
            sampling_rate=np.rint(1/(time_trace[1]-time_trace[0]))
            sampling_rate_line=pd.Series([current_sweep,sampling_rate],index=['Sweep','Sampling_Rate_Hz'])
            
            stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')[-2:]
            
            Stim_start_line=pd.Series([current_sweep,stim_start_time],index=['Sweep','Stim_start_s'])
            Stim_end_line=pd.Series([current_sweep,stim_end_time],index=['Sweep','Stim_end_s'])
            
            Bridge_Error=ephys_treat.estimate_bridge_error(current_TPC, stim_amp, stim_start_time, stim_end_time,do_plot=False)
            Bridge_Error_line=pd.Series([current_sweep,Bridge_Error],index=['Sweep','Bridge_Error_GOhms'])

            if np.isnan(Bridge_Error):
                BE_extrapolated=True
            else:
                BE_extrapolated=False
            BE_extrapolated_line=pd.Series([current_sweep,BE_extrapolated],index=['Sweep','Bridge_Error_extrapolated'])
            
            
            start_index=current_TPC.index[current_TPC['Time_s']<=stim_start_time].to_list()

            start_index=start_index[-1]
            
            membrane_baseline=np.mean(potential_trace[:start_index])
            stim_baseline=np.mean(stim_input_trace[:start_index])
            
            best_A,best_tau,membrane_SS=fitlib.fit_membrane_trace(current_TPC,stim_start_time,stim_end_time,do_plot=False)

            Input_resistance=((membrane_SS-membrane_baseline)/(stim_amp-stim_baseline))*1e3 #Convert to MOhms
            IR_line=pd.Series([current_sweep,Input_resistance],index=['Sweep','Input_Resistance_MOhm'])
            
            
            time_cst=best_tau*1e3 # convert s to ms
            Time_cst_line=pd.Series([current_sweep,time_cst],index=['Sweep','Time_constant_ms'])
            
            Train_df=Train_df.append(train_line,ignore_index=True)
            Trace_df=Trace_df.append(trace_line,ignore_index=True)
            Stim_amp_df=Stim_amp_df.append(stim_amp_line,ignore_index=True)
            Stim_start_df=Stim_start_df.append(Stim_start_line,ignore_index=True)
            Stim_end_df=Stim_end_df.append(Stim_end_line,ignore_index=True)
            Bridge_Error_df=Bridge_Error_df.append(Bridge_Error_line,ignore_index=True)
            Bridge_Error_extrapolated_df=Bridge_Error_extrapolated_df.append(BE_extrapolated_line,ignore_index=True)
            Time_constant_df=Time_constant_df.append(Time_cst_line,ignore_index=True)
            Input_Resistance_df=Input_Resistance_df.append(IR_line,ignore_index=True)
            Sampling_rate_df=Sampling_rate_df.append(sampling_rate_line,ignore_index=True)
            
        
        cell_sweep_info_table= functools.reduce(lambda left, right:     # Merge all pandas DataFrames
                     pd.merge(left , right,
                              on = ["Sweep"]),
                     [Stim_amp_df,Train_df,Trace_df,Stim_start_df,Stim_end_df,Bridge_Error_df,Bridge_Error_extrapolated_df,Time_constant_df,Input_Resistance_df,Sampling_rate_df])
        cell_sweep_info_table=cell_sweep_info_table.sort_values(by=['Train_id','Trace_id','Stim_amp_pA'])
        
        extrapolated_BE = pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms'])

        for current_train in cell_sweep_info_table['Train_id'].unique():
            reduced_cell_sweep_info_table=cell_sweep_info_table[cell_sweep_info_table['Train_id']==current_train]
            BE_array=np.array(reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
            sweep_array=np.array(reduced_cell_sweep_info_table['Sweep'])
            
            
            
            nan_ind_BE = np.isnan(BE_array) 
    
            x = np.arange(len(BE_array))
            if False in nan_ind_BE:
            
                BE_array[nan_ind_BE] = np.interp(x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])
            extrapolated_BE_Series=pd.Series(BE_array)
            sweep_array=pd.Series(sweep_array)
            extrapolated_BE_Series=pd.DataFrame(pd.concat([sweep_array,extrapolated_BE_Series],axis=1))
            extrapolated_BE_Series.columns=['Sweep','Bridge_Error_GOhms']
            extrapolated_BE=extrapolated_BE.append(extrapolated_BE_Series,ignore_index=True)
        
        cell_sweep_info_table.pop('Bridge_Error_GOhms')

        cell_sweep_info_table=cell_sweep_info_table.merge(extrapolated_BE,how='inner', on='Sweep')

        cell_sweep_info_table=cell_sweep_info_table.loc[:,['Sweep',
                                                           "Train_id",
                                                           'Trace_id',
                                                             'Stim_amp_pA',
                                                             'Stim_start_s',
                                                             'Stim_end_s',
                                                             'Bridge_Error_GOhms',
                                                             'Bridge_Error_extrapolated',
                                                             'Time_constant_ms',
                                                             'Input_Resistance_MOhm',
                                                             'Sampling_Rate_Hz']]
                                                                
    cell_sweep_info_table.index=cell_sweep_info_table.loc[:,'Sweep']
    cell_sweep_info_table.index=cell_sweep_info_table.index.astype(int)
    convert_dict = {'Sweep': int,
                    'Train_id': int,
                    'Trace_id': int,
                    'Stim_amp_pA' : float,
                    'Stim_start_s' : float,
                    'Stim_end_s' : float,
                    'Bridge_Error_GOhms' : float,
                    'Bridge_Error_extrapolated' : bool,
                    'Time_constant_ms' : float,
                    'Input_Resistance_MOhm' : float,
                    'Sampling_Rate_Hz' : float
                    }

    cell_sweep_info_table = cell_sweep_info_table.astype(convert_dict)
    return cell_sweep_info_table
            
def get_sweep_info(cell_full_TPC_table,current_sweep,cell_id,database):
    if database=='Allen':
        my_Cell_data = ctc.get_ephys_data(cell_id)
        stim_amp=my_Cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
        
        
        train_id=1
        trace_id=current_sweep
        
        index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
        sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]

        
        
        current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
        current_stim_array=np.array(current_stim_array)
        
        stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
        current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
        
        stim_start_time=current_time_array[stim_start_index]
        stim_end_time=stim_start_time+1.
        

        
        current_TPC=cell_full_TPC_table.loc[current_sweep,'TPC'].copy()
    
        Bridge_Error,membrane_time_cst,R_in,V_rest=ephys_treat.estimate_bridge_error(current_TPC, stim_amp, stim_start_time, stim_end_time,do_plot=False)
    

        if np.isnan(Bridge_Error):
            BE_extrapolated=True
        else:
            BE_extrapolated=False

        
        
        output_line=pd.Series([current_sweep,
                               train_id,
                               trace_id,
                               stim_amp,
                               stim_start_time,
                               stim_end_time,
                               Bridge_Error,
                               BE_extrapolated,
                               membrane_time_cst,
                               R_in,
                               V_rest,
                               sampling_rate],index=['Sweep','Train_id','Trace_id','Stim_amp_pA','Stim_start_s','Stim_end_s','Bridge_Error_GOhms','Bridge_Error_extrapolated',"Time_constant_ms",'Input_Resistance_MOhm','Membrane_resting_potential_mV','Sampling_Rate_Hz'])
        
    elif database == 'Lantyer':
        cell_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=cell_sweep_stim_table[cell_sweep_stim_table['Cell_id']==cell_id]
        sweep_stim_table=pd.DataFrame(cell_sweep_stim_table.iloc[0,2])
        current_TPC=cell_full_TPC_table.loc[current_sweep,'TPC'].copy()
        time_trace=np.array(current_TPC['Time_s'])
      
        train_id=int(str(current_sweep)[0])
        trace_id=int(str(current_sweep)[1:])
        
        stim_amp=np.float64(sweep_stim_table[sweep_stim_table['Sweep_id']==current_sweep]["Stim_amp_pA"].values)
        
        sampling_rate=np.rint(1/(time_trace[1]-time_trace[0]))
        
        stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')[-2:]
        
        Bridge_Error,membrane_time_cst,R_in,V_rest=ephys_treat.estimate_bridge_error(current_TPC, stim_amp, stim_start_time, stim_end_time,do_plot=False)
    

        if np.isnan(Bridge_Error):
            BE_extrapolated=True
        else:
            BE_extrapolated=False

        

        
        output_line=pd.Series([current_sweep,
                               train_id,
                               trace_id,
                               stim_amp,
                               stim_start_time,
                               stim_end_time,
                               Bridge_Error,
                               BE_extrapolated,
                               membrane_time_cst,
                               R_in,
                               V_rest,
                               sampling_rate],index=['Sweep','Train_id','Trace_id','Stim_amp_pA','Stim_start_s','Stim_end_s','Bridge_Error_GOhms','Bridge_Error_extrapolated',"Time_constant_ms",'Input_Resistance_MOhm','Membrane_resting_potential_mV','Sampling_Rate_Hz'])
        
    return output_line


    
def create_cell_sweep_info_table_concurrent_runs(cell_id,database,cell_full_TPC_table):
    cell_sweep_info_table=pd.DataFrame(columns=['Sweep','Train_id','Trace_id','Stim_amp_pA','Stim_start_s','Stim_end_s','Bridge_Error_GOhms','Bridge_Error_extrapolated',"Time_constant_ms",'Input_Resistance_MOhm','Membrane_resting_potential_mV','Sampling_Rate_Hz'])
    sweep_list=np.array(cell_full_TPC_table['Sweep'])
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        sweep_lines_dict={executor.submit(get_sweep_info,cell_full_TPC_table, x,cell_id,database): x for x in sweep_list}
        for f in concurrent.futures.as_completed(sweep_lines_dict):
            cell_sweep_info_table=cell_sweep_info_table.append(f.result(),ignore_index=True)
    
    
    cell_sweep_info_table=cell_sweep_info_table.sort_values(by=['Train_id','Trace_id','Stim_amp_pA'])
    
    extrapolated_BE = pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms'])

    for current_train in cell_sweep_info_table['Train_id'].unique():
        reduced_cell_sweep_info_table=cell_sweep_info_table[cell_sweep_info_table['Train_id']==current_train]
        BE_array=np.array(reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
        sweep_array=np.array(reduced_cell_sweep_info_table['Sweep'])
        
        
        
        nan_ind_BE = np.isnan(BE_array) 

        x = np.arange(len(BE_array))
        if False in nan_ind_BE:
        
            BE_array[nan_ind_BE] = np.interp(x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])
        
        
            
        extrapolated_BE_Series=pd.Series(BE_array)
        sweep_array=pd.Series(sweep_array)
        extrapolated_BE_Series=pd.DataFrame(pd.concat([sweep_array,extrapolated_BE_Series],axis=1))
        extrapolated_BE_Series.columns=['Sweep','Bridge_Error_GOhms']
        extrapolated_BE=extrapolated_BE.append(extrapolated_BE_Series,ignore_index=True)
    
    cell_sweep_info_table.pop('Bridge_Error_GOhms')

    cell_sweep_info_table=cell_sweep_info_table.merge(extrapolated_BE,how='inner', on='Sweep')

    cell_sweep_info_table=cell_sweep_info_table.loc[:,['Sweep',
                                                       "Train_id",
                                                       'Trace_id',
                                                         'Stim_amp_pA',
                                                         'Stim_start_s',
                                                         'Stim_end_s',
                                                         'Bridge_Error_GOhms',
                                                         'Bridge_Error_extrapolated',
                                                         'Time_constant_ms',
                                                         'Input_Resistance_MOhm',
                                                         "Membrane_resting_potential_mV",
                                                         'Sampling_Rate_Hz']]
    
    
    
    cell_sweep_info_table.index=cell_sweep_info_table.loc[:,'Sweep']
    cell_sweep_info_table.index=cell_sweep_info_table.index.astype(int)
    convert_dict = {'Sweep': int,
                    'Train_id': int,
                    'Trace_id': int,
                    'Stim_amp_pA' : float,
                    'Stim_start_s' : float,
                    'Stim_end_s' : float,
                    'Bridge_Error_GOhms' : float,
                    'Bridge_Error_extrapolated' : bool,
                    'Time_constant_ms' : float,
                    'Input_Resistance_MOhm' : float,
                    "Membrane_resting_potential_mV" : float,
                    'Sampling_Rate_Hz' : float
                    }

    cell_sweep_info_table = cell_sweep_info_table.astype(convert_dict)
    return cell_sweep_info_table
            

            
            
            
            
def create_SF_table(TPC_table,SF_dict):
    threshold_table=TPC_table.iloc[SF_dict['Threshold'],:].copy(); threshold_table['Feature']='Threshold'
    peak_table=TPC_table.iloc[SF_dict['Peak'],:].copy(); peak_table['Feature']='Peak'
    upstroke_table=TPC_table.iloc[SF_dict['Upstroke'],:].copy(); upstroke_table['Feature']='Upstroke'    
    downstroke_table=TPC_table.iloc[SF_dict['Downstroke'],:].copy(); downstroke_table['Feature']='Downstroke'
    trough_table=TPC_table.iloc[SF_dict['Trough'],:].copy(); trough_table['Feature']='Trough'
    fast_trough_table=TPC_table.iloc[SF_dict['Fast_Trough'],:].copy(); fast_trough_table['Feature']='Fast_Trough'
    slow_trough_table=TPC_table.iloc[SF_dict['Slow_Trough'],:].copy(); slow_trough_table['Feature']='Slow_Trough'    
    adp_table=TPC_table.iloc[SF_dict['ADP'],:].copy(); adp_table['Feature']='ADP'
    f_AHP_table=TPC_table.iloc[SF_dict['fAHP'],:].copy(); f_AHP_table['Feature']='fAHP'
    
    
    feature_table=pd.concat([threshold_table,peak_table,upstroke_table,downstroke_table,trough_table,fast_trough_table,slow_trough_table,adp_table,f_AHP_table])

    return  feature_table

def create_full_SF_table(original_full_TPC_table,original_full_SF_dict):
    
    full_TPC_table=original_full_TPC_table.copy()
    full_SF_dict=original_full_SF_dict.copy()
    
    sweep_list=np.array(full_TPC_table['Sweep'])
    full_SF_table=pd.DataFrame(columns=["Sweep","SF"])
    
    for current_sweep in sweep_list:
        current_SF_table=create_SF_table(full_TPC_table.loc[current_sweep,'TPC'].copy(),full_SF_dict.loc[current_sweep,'SF_dict'].copy())
        new_line=pd.Series([current_sweep,current_SF_table],index=["Sweep","SF"])
        
        full_SF_table=full_SF_table.append(new_line,ignore_index=True)
    
    full_SF_table.index=full_SF_table['Sweep']
    full_SF_table.index=full_SF_table.index.astype(int)
    
    return full_SF_table



def create_full_TPC_SF_table(cell_id,database):
    full_TPC_table=pd.DataFrame(columns=['Sweep','Bridge_Error_GOhms','TPC'])
    full_SF_table=pd.DataFrame(columns=['Sweep',"Stim_start_s","Stim_end_s",'Stim_amp_pA','SF'])
    if database=='Allen':
        my_Cell_data = ctc.get_ephys_data(cell_id)
        sweep_type=coarse_or_fine_sweep(cell_id)
        sweep_type=sweep_type[sweep_type["stimulus_description"]=="COARSE"]
        coarse_sweep_number=sweep_type.loc[:,"sweep_number"].values
        
        for current_sweep in coarse_sweep_number:
            
            index_range=my_Cell_data.get_sweep(current_sweep)["index_range"]
            
            sampling_rate=my_Cell_data.get_sweep(current_sweep)["sampling_rate"]
            current_stim_array=(my_Cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
            current_membrane_trace=(my_Cell_data.get_sweep(current_sweep)["response"][0:index_range[1]+1])* 1e3 # to mV
            current_membrane_trace=np.array(current_membrane_trace)
            current_stim_array=np.array(current_stim_array)
            stim_start_index=index_range[0]+next(x for x, val in enumerate(current_stim_array[index_range[0]:]) if val != 0 )
            current_time_array=np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate)
            stim_amp=my_Cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
            
            
            stim_start_time=current_time_array[stim_start_index]
            stim_end_time=stim_start_time+1.
            original_TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            Bridge_Error=ephys_treat.estimate_bridge_error(original_TPC, stim_amp, stim_start_time, stim_end_time)
            
            corrected_membrane_trace=current_membrane_trace-Bridge_Error*current_stim_array
            TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=corrected_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            
            # TPC=TPC[TPC['Time_s']<(stim_end_time+.1)]
            # TPC=TPC[TPC['Time_s']>(stim_start_time-.1)]
            TPC.index=TPC.index.astype(int)
            new_line=pd.Series([current_sweep,Bridge_Error,TPC],index=['Sweep','Bridge_Error_GOhms','TPC'])
            full_TPC_table=full_TPC_table.append(new_line,ignore_index=True)
            
            
            current_SF_table=ephys_treat.identify_spike(corrected_membrane_trace, current_time_array, current_stim_array, stim_start_time, stim_end_time,do_plot=False)
            current_SF_table.index=current_SF_table.index.astype(int)
            new_SF_line=pd.Series([current_sweep,stim_start_time,stim_end_time,stim_amp,current_SF_table],index=['Sweep',"Stim_start_s","Stim_end_s",'Stim_amp_pA','SF'])
            full_SF_table=full_SF_table.append(new_SF_line,ignore_index=True)
            
    elif database== 'Lantyer':
        cell_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=cell_sweep_stim_table[cell_sweep_stim_table['Cell_id']==cell_id]
        sweep_list=np.array(cell_sweep_stim_table.iloc[0,-1])
        sweep_stim_table=pd.DataFrame(cell_sweep_stim_table.iloc[0,2])
        
        for current_sweep in sweep_list:
            stim_amp=np.float64(sweep_stim_table[sweep_stim_table['Sweep_id']==current_sweep]["Stim_amp_pA"].values)
            current_time_array,current_membrane_trace,current_stim_array,stim_start_time,stim_end_time=get_traces(cell_id,current_sweep,'Lantyer')
            TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            
            
            #Bridge_Error=ephys_treat.estimate_bridge_error(TPC, stim_amp, stim_start_time, stim_end_time,do_plot=False)
            Bridge_Error=np.nan
            #corrected_membrane_trace=current_membrane_trace-Bridge_Error*current_stim_array
            # TPC=ephys_treat.create_TPC(time_trace=current_time_array,
            #                      potential_trace=corrected_membrane_trace,
            #                      current_trace=current_stim_array,
            #                      filter=2.)
            
            # TPC=TPC[TPC['Time_s']<(stim_end_time+.1)]
            # TPC=TPC[TPC['Time_s']>(stim_start_time-.1)]
            TPC.index=TPC.index.astype(int)
            current_SF_table=ephys_treat.identify_spike(current_membrane_trace, current_time_array, current_stim_array, stim_start_time, stim_end_time,do_plot=False)
            current_SF_table.index=current_SF_table.index.astype(int)
            #current_SF_table=current_SF_table.reset_index()
            
            new_SF_line=pd.Series([current_sweep,stim_start_time,stim_end_time,stim_amp,current_SF_table],index=['Sweep',"Stim_start_s","Stim_end_s",'Stim_amp_pA','SF'])
            full_SF_table=full_SF_table.append(new_SF_line,ignore_index=True)
            new_line=pd.Series([current_sweep,Bridge_Error,TPC],index=['Sweep','Bridge_Error_GOhms','TPC'])
            full_TPC_table=full_TPC_table.append(new_line,ignore_index=True)
    full_TPC_table.index=full_TPC_table.loc[:,'Sweep']

    full_SF_table.index=full_SF_table.loc[:,'Sweep']
    
    return full_TPC_table,full_SF_table
        
def create_metadata_dict(cell_id,database,population_class,original_cell_sweep_info_table):
    metadata_table=pd.DataFrame(columns=["Database",
                          "Area",
                          "Layer",
                          "Dendrite_type",
                          'Nb_of_Traces',
                          'Nb_of_Train'])
    
    population_class.index=population_class.loc[:,"Cell_id"]
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    if database=="Allen":
        cell_data=ctc.get_ephys_features(cell_id) 
        cell_data=cell_data[cell_data["specimen_id"]==cell_id]
        cell_data.index=cell_data.loc[:,'specimen_id']
        
        
        population_class_line=population_class[population_class["Cell_id"]==str(cell_id)]
        Area=population_class.loc[str(cell_id),"General_area"]
        Layer=population_class_line.loc[str(cell_id),"Layer"]
        dendrite_type=population_class_line.loc[str(cell_id),"Dendrite_type"]
        Nb_of_Traces=cell_sweep_info_table.shape[0]
        Nb_of_Train=1
        
    elif database=='Lantyer':
        population_class_line=population_class[population_class["Cell_id"]==str(cell_id)]
        Area=population_class.loc[str(cell_id),"General_area"]
        Layer=population_class_line.loc[str(cell_id),"Layer"]
        dendrite_type=population_class_line.loc[str(cell_id),"Dendrite_type"]
        
        Nb_of_Train=len(cell_sweep_info_table['Train_id'].unique())
        Nb_of_Traces=cell_sweep_info_table.shape[0]
        
      
    new_line=pd.Series([database,
                        Area,
                        Layer,
                        dendrite_type,
                        Nb_of_Traces,
                        Nb_of_Train],index=["Database",
                                              "Area",
                                              "Layer",
                                              "Dendrite_type",
                                              'Nb_of_Traces',
                                              'Nb_of_Train'])
    metadata_table=metadata_table.append(new_line,ignore_index=True)
    metadata_table_dict=metadata_table.loc[0,:].to_dict()
    return metadata_table_dict
            
    
    
    
def create_metadata_table_old(cell_id,database,population_class,full_TPC,full_SF):
    metadata_table=pd.DataFrame(columns=["Database",
                          "Area",
                          "Layer",
                          "Dendrite_type",
                          'Nb_of_Traces',
                          'Nb_of_Protocols',])
    population_class.index=population_class.loc[:,"Cell_id"]
    if database=="Allen":
        cell_data=ctc.get_ephys_features(cell_id) 
        cell_data=cell_data[cell_data["specimen_id"]==cell_id]
        cell_data.index=cell_data.loc[:,'specimen_id']
       
        IR_computed,IR_computed_sd,Time_cst_computed,Time_cst_computed_sd=ephys_treat.compute_input_resistance_time_cst(full_TPC,full_SF)
        

        population_class_line=population_class[population_class["Cell_id"]==str(cell_id)]
        Area=population_class.loc[str(cell_id),"General_area"]
        Layer=population_class_line.loc[str(cell_id),"Layer"]
        dendrite_type=population_class_line.loc[str(cell_id),"Dendrite_type"]
        Nb_of_Traces=full_SF.shape[0]
        Nb_of_Protocols=1
        
    
    
    elif database=='Lantyer':
        population_class_line=population_class[population_class["Cell_id"]==str(cell_id)]
        Area=population_class.loc[str(cell_id),"General_area"]
        Layer=population_class_line.loc[str(cell_id),"Layer"]
        dendrite_type=population_class_line.loc[str(cell_id),"Dendrite_type"]
        
        IR_computed,IR_computed_sd,Time_cst_computed,Time_cst_computed_sd=ephys_treat.compute_input_resistance_time_cst(full_TPC,full_SF)
        Nb_of_Traces=full_SF.shape[0]
        for elt in [3,2,1]:
            if Nb_of_Traces % elt == 0:
                Nb_of_Protocols=elt
                break
      
        
        
    new_line=pd.Series([database,
                        Area,
                        Layer,
                        dendrite_type,
                        Nb_of_Traces,
                        Nb_of_Protocols,],index=["Database",
                                              "Area",
                                              "Layer",
                                              "Dendrite_type",
                                              'Nb_of_Traces',
                                              'Nb_of_Protocols'])
    metadata_table=metadata_table.append(new_line,ignore_index=True)
    return metadata_table
        


def process_cell_data(cell_id,database,population_class_file):
    
    if database == 'Allen':
        cell_id=int(cell_id)

    full_TPC_table=create_cell_full_TPC_table_concurrent_runs(cell_id, database)

    cell_sweep_info_table=create_cell_sweep_info_table_concurrent_runs(cell_id, database, full_TPC_table.copy())

    full_SF_dict=create_cell_full_SF_dict_table(full_TPC_table.copy(), cell_sweep_info_table.copy())

    full_SF_table=create_full_SF_table(full_TPC_table.copy(),full_SF_dict.copy())

    #TPC,SF_table=create_full_TPC_SF_table(cell_id,database)
    
    
    
    time_response_list=[.005,.010,.025,.050,.100,.250,.500]
    
    fit_columns=['Response_time_ms','I_O_obs','I_O_QNRMSE','Hill_amplitude','Hill_coef','Hill_Half_cst','Adaptation_obs','Adaptation_RMSE','A','B','C']
    cell_fit_table=pd.DataFrame(columns=fit_columns)
    
    feature_columns=['Response_time_ms','Gain','Threshold','Saturation','Adaptation_index']
    cell_feature_table=pd.DataFrame(columns=feature_columns)
    
    Full_Adaptation_fit_table_columns=['Response_time_ms',"Sweep","Stim_amp_pA","A",'B','C','Nb_of_spikes','RMSE']
    Full_Adaptation_fit_table=pd.DataFrame(columns=Full_Adaptation_fit_table_columns)
    
    

    for response_time in time_response_list:
        print('Respinse_time',response_time)
        stim_freq_table=fitlib.get_stim_freq_table(full_SF_table.copy(),cell_sweep_info_table.copy(),response_time)
        #return stim_freq_table
        pruning_obs,do_fit=data_pruning(stim_freq_table)
        if do_fit ==True:
            I_O_obs,Hill_Amplitude,Hill_coef,Hill_Half_cst,I_O_QNRMSE,x_shift,Gain,Threshold,Saturation=fitlib.fit_IO_curve(stim_freq_table,do_plot=False)
        else:
            I_O_obs=pruning_obs
            empty_array=np.empty(8)
            empty_array[:]=np.nan
            Hill_Amplitude,Hill_coef,Hill_Half_cst,I_O_QNRMSE,x_shift,Gain,Threshold,Saturation=empty_array
            
        adapt_obs,adapt_do_fit=data_pruning_adaptation(full_SF_table,cell_sweep_info_table.copy(), response_time)

        if adapt_do_fit==True:
            inst_freq_table=fitlib.extract_inst_freq_table(full_SF_table,cell_sweep_info_table.copy(),response_time)
            Adaptation_obs,A,Adaptation_index,C,Adaptation_RMSE,Adaptation_table=fitlib.fit_adaptation_curve(inst_freq_table,do_plot=False)
            
        else:
            Adaptation_obs=adapt_obs
            empty_array=np.empty(4)
            empty_array[:]=np.nan
            A,Adaptation_index,C,Adaptation_RMSE=empty_array
        new_fit_table_line=pd.Series([response_time*1e3,
                                      I_O_obs,
                                      I_O_QNRMSE,
                                      Hill_Amplitude,
                                      Hill_coef,
                                      Hill_Half_cst,
                                      Adaptation_obs,
                                      Adaptation_RMSE,
                                      A,
                                      Adaptation_index,
                                      C],index=fit_columns)
        
        cell_fit_table=cell_fit_table.append(new_fit_table_line,ignore_index=True)
        
        new_feature_table_line=pd.Series([response_time*1e3,
                                          Gain,
                                          Threshold,
                                          Saturation,
                                          Adaptation_index],index=feature_columns)
        
        cell_feature_table=cell_feature_table.append(new_feature_table_line,ignore_index=True)
        
        if Adaptation_obs=='--':
            Adaptation_table['Cell_id']=str(cell_id)
            Adaptation_table['Response_time_ms']=response_time*1e3
            
            Adaptation_table=Adaptation_table.loc[:,['Response_time_ms',"Sweep","Stim_amp_pA","A",'B','C','Nb_of_spikes','RMSE']]

            #Full_Adaptation_fit_table=Full_Adaptation_fit_table.append(Adaptation_table,ignore_index=True)
            
            Full_Adaptation_fit_table=pd.concat([Full_Adaptation_fit_table, Adaptation_table], axis=0)
            

    Full_Adaptation_fit_table.index=np.arange(Full_Adaptation_fit_table.shape[0])
    cell_fit_table.index=cell_fit_table.loc[:,"Response_time_ms"]
    cell_feature_table.index=cell_feature_table.loc[:,'Response_time_ms']
    cell_feature_table.index=cell_feature_table.index.astype(int)

    metadata_table=create_metadata_dict(cell_id,database,population_class_file,cell_sweep_info_table.copy())
    #metadata_table.index=metadata_table.index.astype(int)
    full_TPC_table.index=full_TPC_table.index.astype(int)
    full_SF_table.index=full_SF_table.index.astype(int)
    
    if cell_fit_table.shape[0]==0:
        cell_fit_table.loc[0,:]=np.nan
        cell_fit_table=cell_fit_table.apply(pd.to_numeric)
    if cell_feature_table.shape[0]==0:
        cell_feature_table.loc[0,:]=np.nan
        cell_feature_table=cell_feature_table.apply(pd.to_numeric)
    if Full_Adaptation_fit_table.shape[0]==0:
        Full_Adaptation_fit_table.loc[0,:]=np.nan
        Full_Adaptation_fit_table=Full_Adaptation_fit_table.apply(pd.to_numeric)
    
    
    
    return metadata_table,full_TPC_table,full_SF_table,full_SF_dict,cell_sweep_info_table,cell_fit_table,cell_feature_table,Full_Adaptation_fit_table
    
        
def write_cell_file_h5(cell_file_path,original_full_TPC_table,original_full_SF_dict,original_cell_sweep_info_table,original_cell_fit_table,original_cell_feature_table,original_Full_Adaptation_fit_table,original_metadata_table):
    full_TPC_table=original_full_TPC_table.copy()
    full_SF_dict=original_full_SF_dict.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    cell_fit_table=original_cell_fit_table.copy()
    cell_feature_table = original_cell_feature_table.copy()
    adaptation_fit_table = original_Full_Adaptation_fit_table.copy()
    
    #f=h5py.File("/Users/julienballbe/Downloads/New_cell_file_full_test.h5", "w")
    f=h5py.File(cell_file_path, "w")
    
    ### Store Metadata Table
    Metadata_group=f.create_group('Metadata_table')
    for elt in original_metadata_table.keys():
        

        Metadata_group.create_dataset(str(elt),data=original_metadata_table[elt])
    
    
    ### Store Cell Sweep Info Table
    cell_sweep_info_table_group=f.create_group('Sweep_info')
    for elt in np.array(cell_sweep_info_table.columns):
        cell_sweep_info_table_group.create_dataset(elt,data=np.array(cell_sweep_info_table[elt]))
       
    sweep_list=np.array(full_TPC_table['Sweep'])
    
   
    
    ### Store TPC Tables and SF dict
    TPC_group=f.create_group('TPC_tables')
    SF_group=f.create_group("SF_tables")
    TPC_colnames=np.array(full_TPC_table.loc[sweep_list[0],"TPC"].columns)
    TPC_group.create_dataset("TPC_colnames",data=TPC_colnames)
    for current_sweep in sweep_list:
        TPC_group.create_dataset(str(current_sweep),data=full_TPC_table.loc[current_sweep,"TPC"])
        
        current_SF_dict=full_SF_dict.loc[current_sweep,"SF_dict"]
        current_SF_group=SF_group.create_group(str(current_sweep))
        for elt in current_SF_dict.keys():
            
            if len(current_SF_dict[elt])!=0:
                current_SF_group.create_dataset(str(elt),data=current_SF_dict[elt])
    
    ### Store Cell Feature Table
    cell_feature_group = f.create_group('Cell_Feature')
    cell_feature_group.create_dataset('Cell_Feature_Table',data = cell_feature_table)
    cell_feature_group.create_dataset('Cell_Feature_Table_colnames',data = np.array(cell_feature_table.columns))
    
    ###Store Cell fit table
    
    cell_fit_group = f.create_group('Cell_Fit')
    
    for elt in np.array(cell_fit_table.columns):
        cell_fit_group.create_dataset(elt,data=np.array(cell_fit_table[elt]))
       
    sweep_list=np.array(full_TPC_table['Sweep'])
    
    ### Store Adaptation fit table
    Adaptation_group = f.create_group("Adaptation_Fit")
    Adaptation_group.create_dataset('Adaptation_Fit_Table',data = adaptation_fit_table)
    Adaptation_group.create_dataset('Adaptation_Fit_Table_colnames',data = np.array(adaptation_fit_table.columns))
    
    
    f.close()
    
def create_cell_file(cell_id_list,saving_folder_path = "/Volumes/Work_Julien/Cell_Data_File/",overwrite=False):
    
    today=date.today()
    today=today.strftime("%Y_%m_%d")
    original_files=[f for f in glob.glob(str(saving_folder_path+'*.h5*'))]
    population_class_table=pd.read_csv("/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Population_Class.csv")
    population_class_table=population_class_table.copy()
    full_database_table=pd.read_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Database_Cell_id.csv')
    full_database_table=full_database_table.copy()
    
    

    for cell_id in tqdm(cell_id_list):
        Cell_file_information=pd.read_csv("/Volumes/Work_Julien/Cell_Data_File/Cell_file_information.csv")
        Cell_file_information=Cell_file_information.copy()
        cell_file_path=str(saving_folder_path+'Cell_'+str(cell_id)+'_data_file.h5')
        cell_index=Cell_file_information[Cell_file_information['Cell_id']==str(cell_id)].index.tolist()
        try:
            if cell_file_path in original_files and overwrite ==  True:
                
                
                cell_database=full_database_table[full_database_table['Cell_id']==str(cell_id)]['Database'].values[0]
                metadata_table,full_TPC_table,full_SF_table,full_SF_dict,cell_sweep_info_table,cell_fit_table,cell_feature_table,Full_Adaptation_fit_table=process_cell_data(str(cell_id),cell_database,population_class_table)
    
                os.remove(cell_file_path)
                write_cell_file_h5(cell_file_path,full_TPC_table,full_SF_dict,cell_sweep_info_table,cell_fit_table,cell_feature_table,Full_Adaptation_fit_table,metadata_table)
                
                Cell_file_information.loc[cell_index,'Note']='--'
                Cell_file_information.loc[cell_index,'Lastly_modified']=str(today)
                Cell_file_information.loc[cell_index,'Created']=str(today)
                
                
                
            elif cell_file_path in original_files and overwrite ==  False:
                print( str('File for cell '+str(cell_id)+' already existing'))
            
            else:
                
                cell_database=full_database_table[full_database_table['Cell_id']==str(cell_id)]['Database'].values[0]
                
                metadata_table,full_TPC_table,full_SF_table,full_SF_dict,cell_sweep_info_table,cell_fit_table,cell_feature_table,Full_Adaptation_fit_table=process_cell_data(str(cell_id),cell_database,population_class_table)
                

                write_cell_file_h5(cell_file_path,full_TPC_table,full_SF_dict,cell_sweep_info_table,cell_fit_table,cell_feature_table,Full_Adaptation_fit_table,metadata_table)
                Cell_file_information.loc[cell_index,'Note']='--'
                Cell_file_information.loc[cell_index,'Lastly_modified']=str(today)
                Cell_file_information.loc[cell_index,'Created']=str(today)
            
        except Exception as e:
                
                print(str('Problem_with_cell:'+str(cell_id)+str(e)))
                Cell_file_information.loc[cell_index,'Note']=str(type(e))
                
        Cell_file_information=Cell_file_information.loc[:,['Cell_id','Database','Note','Created','Lastly_modified']]
        Cell_file_information.to_csv("/Volumes/Work_Julien/Cell_Data_File/Cell_file_information.csv")
    
    return 'Done'
    
    
    
def import_cell_json_file(file_path):
    full_json_file=pd.read_json(file_path)
    full_TPC_table=pd.DataFrame(full_json_file["TPC"][0])
    full_TPC_table.index=full_TPC_table.index.astype(int)
    
    full_SF_table=pd.DataFrame(full_json_file["Spike_feature_table"][0])
    full_SF_table.index=full_SF_table.index.astype(int)
    
    Metadata_table=pd.DataFrame(full_json_file["Metadata"][0])
    Fit_table=pd.DataFrame(full_json_file["Fit_table"][0])
    IO_table=pd.DataFrame(full_json_file["IO_table"][0])
    Adaptation_fit_table=pd.DataFrame(full_json_file['Adaptation_fit_table'][0])
    
    return full_SF_table,full_TPC_table,Metadata_table,Fit_table,IO_table,Adaptation_fit_table

    
def import_cell_pkl_file(file_path):
    full_pkl_file=pd.read_pickle(file_path)
    full_TPC_table=pd.DataFrame(full_pkl_file["TPC"][0])
    full_TPC_table.index=full_TPC_table.index.astype(int)
    
    full_SF_table=pd.DataFrame(full_pkl_file["Spike_feature_table"][0])
    full_SF_table.index=full_SF_table.index.astype(int)
    
    Metadata_table=pd.DataFrame(full_pkl_file["Metadata"][0])
    Fit_table=pd.DataFrame(full_pkl_file["Fit_table"][0])
    IO_table=pd.DataFrame(full_pkl_file["IO_table"][0])
    Adaptation_fit_table=pd.DataFrame(full_pkl_file['Adaptation_fit_table'][0])
    
    return full_SF_table,full_TPC_table,Metadata_table,Fit_table,IO_table,Adaptation_fit_table

def import_cell_hdf_file(file_path):
    full_hdf_file=pd.read_hdf(file_path,'df')
    full_TPC_table=pd.DataFrame(full_hdf_file["TPC"][0])
    full_TPC_table.index=full_TPC_table.index.astype(int)
    
    full_SF_table=pd.DataFrame(full_hdf_file["Spike_feature_table"][0])
    full_SF_table.index=full_SF_table.index.astype(int)
    
    Metadata_table=pd.DataFrame(full_hdf_file["Metadata"][0])
    Fit_table=pd.DataFrame(full_hdf_file["Fit_table"][0])
    IO_table=pd.DataFrame(full_hdf_file["IO_table"][0])
    Adaptation_fit_table=pd.DataFrame(full_hdf_file['Adaptation_fit_table'][0])
    
    return full_SF_table,full_TPC_table,Metadata_table,Fit_table,IO_table,Adaptation_fit_table
    

def get_traces(cell_id,sweep_id,database):
    
    if database=='Allen':
        my_Cell_data = ctc.get_ephys_data(cell_id)
        index_range=my_Cell_data.get_sweep(sweep_id)["index_range"]
        sampling_rate=my_Cell_data.get_sweep(sweep_id)["sampling_rate"]
        current_trace=(my_Cell_data.get_sweep(sweep_id)["stimulus"][0:index_range[1]+1])* 1e12 #to pA
        potential_trace=(my_Cell_data.get_sweep(sweep_id)["response"][0:index_range[1]+1])* 1e3 # to mV
        potential_trace=np.array(potential_trace)
        current_trace=np.array(current_trace)
        stim_start_index=index_range[0]+next(x for x, val in enumerate(current_trace[index_range[0]:]) if val != 0 )
        time_trace=np.arange(0, len(current_trace)) * (1.0 / sampling_rate)
        stim_start_time=time_trace[stim_start_index]
        stim_end_time=stim_start_time+1.
        
    elif database=='Lantyer':
        
        sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Lantyer_Data/ID_Sweep_stim_table.pkl')

        sweep_stim_table=sweep_stim_table[sweep_stim_table['Cell_id']==cell_id]
        file_name=sweep_stim_table.iloc[0,0]
        sweep_stim_table=pd.DataFrame(sweep_stim_table.iloc[0,2])

        cell,sweep,stim_amp,stim_start_time,stim_end_time=sweep_stim_table[sweep_stim_table['Sweep_id']==sweep_id].iloc[0,:]
        
        current_file=scipy.io.loadmat(str('/Users/julienballbe/My_Work/Lantyer_Data/Original_matlab_file/'+str(file_name)))
        
        current_unique_cell_id=cell_id[-1]
        if len(str(sweep))==2:
            current_Train=str(sweep)[0]
            current_sweep=str(sweep)[1]
        elif len(str(sweep))==3:
            current_Train=str(sweep)[0]
            current_sweep=str(sweep)[1:]
            
        current_id=str('Trace_'+current_unique_cell_id+'_'+current_Train+'_'+str(current_sweep))
        
        current_stim_trace= pd.DataFrame(current_file[str(current_id+'_1')],columns=['Time_s','Stimulus_amplitude_pA'])
        current_stim_trace.loc[:,'Stimulus_amplitude_pA']*=1e12
        
        current_membrane_trace= pd.DataFrame(current_file[str(current_id+'_2')],columns=['Time_s','Membrane_potential_mV'])
        current_membrane_trace.loc[:,'Membrane_potential_mV']*=1e3
        
        time_trace=np.array(current_membrane_trace.loc[:,'Time_s'])
        potential_trace=np.array(current_membrane_trace.loc[:,'Membrane_potential_mV'])
        current_trace=np.array(current_stim_trace.loc[:,'Stimulus_amplitude_pA'])
        
    return time_trace,potential_trace,current_trace,stim_start_time,stim_end_time

#%% In process

def identify_sweep_for_first_ap_waveform(cell_id,original_full_SF_table, original_metadata_table):
    """ Find lowest-amplitude spiking sweep that has at least min_spike
        or else sweep with most spikes

        Parameters
        ----------
        sweeps: SweepSet
            Sweeps to consider for ISI shape calculation
        features: dict
            Output of LongSquareAnalysis.analyze()
        duration: float
            Length of stimulus interval (seconds)
        min_spike: int (optional, default 5)
            Minimum number of spikes for first preference sweep (default 5)

        Returns
        -------
        selected_sweep: Sweep
            Sweep object for ISI shape calculation
        selected_spike_info: DataFrame
            Spike info for selected sweep
    """
    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike
    
    full_SF_table=original_full_SF_table.copy()

    metadata_table=original_metadata_table.copy()
    
    if metadata_table.loc[0,'Database']=='Allen':
        ctc=CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        allen_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Allen_Data/Allen_ID_Sweep_stim_table.pkl')
        cell_data=ctc.get_ephys_data(cell_id)

        
        ss_coarse_sweep_table=coarse_or_fine_sweep(cell_id,stimulus_type='Short Square')
        ss_coarse_sweep_table=ss_coarse_sweep_table[ss_coarse_sweep_table['stimulus_description']=='COARSE']
        ss_coarse_sweep_table.index=ss_coarse_sweep_table['sweep_number']
        ss_coarse_sweep_table['Stim_amp_pA']=0
        ss_coarse_sweep_table['Nb_of_Spike']=0
        for current_sweep in np.array(ss_coarse_sweep_table['sweep_number']):
            stim_amp=cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
            ss_coarse_sweep_table.loc[current_sweep,'Stim_amp_pA']=stim_amp
            nb_of_spike=len(cell_data.get_spike_times(current_sweep))
            ss_coarse_sweep_table.loc[current_sweep,'Nb_of_Spike']=nb_of_spike
        ss_coarse_sweep_table=ss_coarse_sweep_table[ss_coarse_sweep_table['Nb_of_Spike']!=0]
        sweep_list=np.array(ss_coarse_sweep_table.sort_values(by=['Stim_amp_pA','Nb_of_Spike'])['sweep_number'])
        first_ap_ss_sweep=sweep_list[0]
        
        ramp_sweep_list=np.array(allen_sweep_stim_table.loc[cell_id,'Ramp'])
        min_ramp_spike_stim=np.inf
        first_ap_ramp_sweep=ramp_sweep_list[0]
        for current_ramp_sweep in ramp_sweep_list:
            first_spike_times=cell_data.get_spike_times(current_ramp_sweep)[0]
            index_range=cell_data.get_sweep(current_ramp_sweep)["index_range"]
            sampling_rate=cell_data.get_sweep(current_ramp_sweep)["sampling_rate"]
            current_stim_array=(cell_data.get_sweep(current_ramp_sweep)["stimulus"][0:index_range[1]+1])* 1e12 
            current_time_array=(np.arange(0, len(current_stim_array)) * (1.0 / sampling_rate))

            stim_time_table=pd.concat([pd.Series(current_stim_array),pd.Series(current_time_array)],axis=1)
            stim_time_table.columns=["Stim_amp_pA","Time_s"]
            
            current_min_ramp_spike_stim=stim_time_table[stim_time_table['Time_s']==first_spike_times]['Stim_amp_pA'].values
            if current_min_ramp_spike_stim<min_ramp_spike_stim:
                min_ramp_spike_stim=current_min_ramp_spike_stim
                first_ap_ramp_sweep=current_ramp_sweep
                
        first_ap_ss_sweep=np.array([first_ap_ss_sweep])
        first_ap_ramp_sweep=np.array([first_ap_ramp_sweep])   
                                         
    else:
        first_ap_ss_sweep=np.array([])
        first_ap_ramp_sweep=np.array([])   
            
    
    sweep_list=np.array(full_SF_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocols'].values[0])
    Nb_of_Traces=full_SF_table.shape[0]
    
    
    first_ap_ls_sweep=np.array([])
    for elt in range(Nb_of_Traces//Nb_of_Protocols):
        current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
        sweep_zero_spike_check=1
        #current_max_nb_of_spike=0
        for current_idx in current_sweeps_idx:
            current_sweep=sweep_list[current_idx]
            current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            current_SF=current_SF[current_SF['Feature']=='Threshold']
            current_SF=current_SF.sort_values(by=['Time_s'])
            current_spike_time = np.array(current_SF['Time_s'])
            nb_of_spike=current_SF[current_SF['Feature']=='Threshold'].shape[0]
            
            current_end_stim_time=full_SF_table.loc[current_sweep,'Stim_end_s']
            
            if current_spike_time.size==0:
                sweep_zero_spike_check*=current_spike_time.size
            
            elif current_spike_time[0]+0.003 < current_end_stim_time: #check if the 3ms window after threshold is not cut by stim end
                sweep_zero_spike_check*=current_spike_time.size
                #current_min_nb_of_spike=min(current_max_nb_of_spike,nb_of_spike)
                
            elif current_spike_time[0]+0.003 >= current_end_stim_time:
                sweep_zero_spike_check*=0 # if the 3ms window after threshold is cut by stim end, do not consider the spike
                
        
        if sweep_zero_spike_check>=1: #in case multiple sweeps have the same stim amplitude, check that each sweep has at least one spike
            first_ap_ls_sweep=sweep_list[current_sweeps_idx]
            break
        
    
    return first_ap_ss_sweep,first_ap_ls_sweep,first_ap_ramp_sweep

def first_ap_vectors(cell_id ,sweeps_list, protocol='ss' ,original_Full_SF_table=None, original_Full_TPC_table=None,
        target_sampling_rate=50000, window_length=0.003,
        skip_clipped=False):
    """Average waveforms of first APs from sweeps

    Parameters
    ----------
    sweeps_list: list
        List of Sweep objects
    spike_info_list: list
        List of spike info DataFrames
    target_sampling_rate: float (optional, default 50000)
        Desired sampling rate of output (Hz)
    window_length: float (optional, default 0.003)
        Length of AP waveform (seconds)

    Returns
    -------
    ap_v: array of shape (target_sampling_rate * window_length)
        Waveform of average AP
    ap_dv: array of shape (target_sampling_rate * window_length - 1)
        Waveform of first derivative of ap_v
    """
    if original_Full_SF_table is not None:
        full_SF_table=original_Full_SF_table.copy()
        full_TPC_table=original_Full_TPC_table.copy()
        
    # if skip_clipped:
    #     nonclipped_sweeps_list = []
    #     nonclipped_spike_info_list = []
    #     for swp, si in zip(sweeps_list, spike_info_list):
    #         if not si["clipped"].values[0]:
    #             nonclipped_sweeps_list.append(swp)
    #             nonclipped_spike_info_list.append(si)
    #     sweeps_list = nonclipped_sweeps_list
    #     spike_info_list = nonclipped_spike_info_list

    if len(sweeps_list) == 0:
        length_in_points = int(target_sampling_rate * window_length)
        zero_v = np.zeros(length_in_points)
        return zero_v, np.diff(zero_v)
    
    if protocol == 'ss' or protocol == "ramp":

        ctc=CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        
        cell_data=ctc.get_ephys_data(cell_id)
        ap_list = []
        for current_sweep in sweeps_list:
            
            first_spike_times=cell_data.get_spike_times(current_sweep)[0]
            index_range=cell_data.get_sweep(current_sweep)["index_range"]
            current_sampling_rate=cell_data.get_sweep(current_sweep)["sampling_rate"]
            current_stim_array=(cell_data.get_sweep(current_sweep)["stimulus"][0:index_range[1]+1])* 1e12 
            time_array=(np.arange(0, len(current_stim_array)) * (1.0 / current_sampling_rate))
            current_response_array=(cell_data.get_sweep(current_sweep)["response"][0:index_range[1]+1])* 1e3

            stim_time_table=pd.concat([pd.Series(current_stim_array),pd.Series(time_array),pd.Series(current_response_array)],axis=1)
            stim_time_table.columns=["Stim_amp_pA","Time_s",'Membrane_potential_mV']
            
            stim_time_table=stim_time_table[stim_time_table["Time_s"]>=first_spike_times]
            stim_time_table=stim_time_table[stim_time_table["Time_s"]<=(first_spike_times+window_length)]
            
            current_time_array=np.array(stim_time_table['Time_s'])
            current_potential_trace=np.array(stim_time_table['Membrane_potential_mV'])
            
            sampling_rate = int(np.rint(1. / (current_time_array[1] - current_time_array[0])))
            length_in_points = int(sampling_rate * window_length)
            
            first_threshold_time=np.array(cell_data.get_spike_times(current_sweep))[0]
            
            ap = first_ap_waveform(current_potential_trace, current_time_array, first_threshold_time, length_in_points)
            ap_list.append(ap)
            
            
        avg_ap = np.vstack(ap_list).mean(axis=0) #Do the average of first AP if there is multiple sweeps at the same stimulus amplitude   
        
        if sampling_rate > target_sampling_rate:
            sampling_factor = int(sampling_rate // target_sampling_rate)
            avg_ap = _subsample_average(avg_ap, sampling_factor)    
            
            
    elif protocol == 'ls':
    
        current_SF_table=pd.DataFrame(full_SF_table.loc[sweeps_list[0],'SF'])
        current_TPC_table=pd.DataFrame(full_TPC_table.loc[sweeps_list[0],'TPC'])
        current_time_array=np.array(current_TPC_table['Time_s'])
        
        
        sampling_rate = int(np.rint(1. / (current_time_array[1] - current_time_array[0])))
        length_in_points = int(sampling_rate * window_length)
        print('sampling_Rate=',sampling_rate)
        
        ap_list = []
        for current_sweep in sweeps_list:
            current_SF_table=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            current_TPC_table=pd.DataFrame(full_TPC_table.loc[current_sweep,'TPC'])
            current_time_array=np.array(current_TPC_table['Time_s'])
            current_potential_trace=np.array(current_TPC_table['Membrane_potential_mV'])
            
            threshold_table=current_SF_table[current_SF_table['Feature']=='Threshold'].sort_values(by=['Time_s'])
            first_threshold_time=np.array(threshold_table['Time_s'])
            first_threshold_time=first_threshold_time[0]
            ap = first_ap_waveform(current_potential_trace, current_time_array, first_threshold_time, length_in_points)
            ap_list.append(ap)
            plt.plot(ap)
            plt.show()
            
        avg_ap = np.vstack(ap_list).mean(axis=0) #Do the average of first AP if there is multiple sweeps at the same stimulus amplitude   
        
        if sampling_rate > target_sampling_rate:
            sampling_factor = int(sampling_rate // target_sampling_rate)
            avg_ap = _subsample_average(avg_ap, sampling_factor)

    return avg_ap, np.diff(avg_ap)

def first_ap_waveform(potential_trace, time_trace, first_threshold_time, length_in_points):
    """Waveform of first AP with `length_in_points` time samples

    Parameters
    ----------
    sweep: Sweep
        Sweep object with spikes
    spikes: DataFrame
        Spike info dataframe with "threshold_index" column
    length_in_points: int
        Length of returned AP waveform

    Returns
    -------
   first_ap_v: array of shape (length_in_points)
        The waveform of the first AP in `sweep`
    """
    start_index=find_time_index(time_trace, first_threshold_time)
    end_index = start_index + length_in_points
    return potential_trace[start_index:end_index]

def _subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg

def create_first_AP_ss_ls_ramp_vector(cell_id , original_full_SF_table,original_full_TPC_table,
                                      original_metadata_table,target_sampling_rate=50000, window_length=0.003,do_plot=False):
    
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table = original_full_TPC_table.copy()
    metadata_table = original_metadata_table.copy()
    
    ss_sweep,ls_sweep,ramp_sweep=identify_sweep_for_first_ap_waveform(cell_id,full_SF_table, metadata_table)
    
    first_ap=np.array([])
    
    
    first_ap_diff=np.array([])
    
    
    ss_ap_wf,ss_ap_wf_diff=first_ap_vectors(cell_id,ss_sweep,protocol='ss',target_sampling_rate=target_sampling_rate,window_length=window_length)
    first_ap=np.append(first_ap,ss_ap_wf)

    first_ap_diff=np.append(first_ap_diff,ss_ap_wf_diff)
    
    ls_ap_wf,ls_ap_wf_diff=first_ap_vectors(cell_id,ls_sweep, protocol= 'ls', original_Full_SF_table = full_SF_table, original_Full_TPC_table = full_TPC_table,target_sampling_rate=target_sampling_rate,window_length=window_length)
    first_ap=np.append(first_ap,ls_ap_wf)
    
    first_ap_diff=np.append(first_ap_diff,ls_ap_wf_diff)
    
    ramp_ap_wf,ramp_ap_wf_diff=first_ap_vectors(cell_id,ramp_sweep, protocol= 'ramp',target_sampling_rate=target_sampling_rate,window_length=window_length)
    first_ap=np.append(first_ap,ramp_ap_wf)

    first_ap_diff=np.append(first_ap_diff,ramp_ap_wf_diff)

    
    
    if do_plot:
        
        
        sampling_rate=int(np.rint(1. / (window_length/len(ss_ap_wf))))
        time_array=np.arange(0, len(first_ap)) * (1.0 / sampling_rate)
        
        time_array_diff=np.arange(0, len(first_ap_diff)) * (1.0 / sampling_rate)
        x1=time_array_diff[len(ss_ap_wf)]
        x2=time_array_diff[len(ss_ap_wf)+len(ls_ap_wf)]
        
        ap_wf_table=pd.concat([pd.Series(first_ap),pd.Series(time_array)],axis=1)
        ap_wf_table.columns=["Membrane_potential_mV","Time_s"]
        
        ap_wf_diff_table=pd.concat([pd.Series(first_ap_diff),pd.Series(time_array_diff)],axis=1)
        ap_wf_diff_table.columns=["Membrane_potential_derivative_mV/s","Time_s"]
        
        
        first_ap_plot=ggplot(ap_wf_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        first_ap_plot+=ggtitle('First_AP_waveform')
        first_ap_plot+=geom_vline(xintercept=[x1,x2],colour=['red','red'])
        print(first_ap_plot)
        
        first_ap_diff_plot=ggplot(ap_wf_diff_table,aes(x='Time_s',y="Membrane_potential_derivative_mV/s"))+geom_line()
        first_ap_diff_plot+=ggtitle('First_AP__derivative_waveform')
        first_ap_diff_plot+=geom_vline(xintercept=[x1,x2],colour=['red','red'])
        print(first_ap_diff_plot)
        
    return first_ap, first_ap_diff
        
    #To be finished; deal with short square sweeps and ramp sweeps in first_ap_vectors

#%% ISI voltage trajectory

def identify_sweep_for_isi_shape(original_full_SF_table, original_metadata_table,duration, min_spike=5):
    """ Find lowest-amplitude spiking sweep that has at least min_spike
        or else sweep with most spikes

        Parameters
        ----------
        sweeps: SweepSet
            Sweeps to consider for ISI shape calculation
        features: dict
            Output of LongSquareAnalysis.analyze()
        duration: float
            Length of stimulus interval (seconds)
        min_spike: int (optional, default 5)
            Minimum number of spikes for first preference sweep (default 5)

        Returns
        -------
        selected_sweep: Sweep
            Sweep object for ISI shape calculation
        selected_spike_info: DataFrame
            Spike info for selected sweep
    """
    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike
    
    full_SF_table=original_full_SF_table.copy()

    metadata_table=original_metadata_table.copy()
    
    sweep_list=np.array(full_SF_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocols'].values[0])
    Nb_of_Traces=full_SF_table.shape[0]
    max_nb_spike=0
    for sweep in sweep_list:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[sweep,'SF'])

        max_nb_spike=max(max_nb_spike,sub_SF_table[sub_SF_table['Feature']=='Peak'].shape[0])
    
    selected_sweep=[]

    for elt in range(Nb_of_Traces//Nb_of_Protocols):
        current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
        current_max_nb_of_spike=0
        for current_idx in current_sweeps_idx:
            current_sweep=sweep_list[current_idx]
            current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
            current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
        
        if max_nb_spike>=min_spike:
            if current_max_nb_of_spike>=5:
                selected_sweep=np.array(sweep_list[current_sweeps_idx])
                break
        elif max_nb_spike>1:
            if current_max_nb_of_spike==max_nb_spike:
                selected_sweep=np.array(sweep_list[current_sweeps_idx])
                break
        else:
            if current_max_nb_of_spike==1:
                selected_sweep=np.array(sweep_list[current_sweeps_idx])
                break
           
    return selected_sweep

def isi_shape(original_SF_table,original_TPC_table, end, n_points=100, steady_state_interval=0.1,
    single_return_tolerance=1., single_max_duration=0.1):
    """ Average interspike voltage trajectory with normalized duration, aligned to threshold, in a given sweep

        Parameters
        ----------
        sweep: Sweep
            Sweep object with at least one action potential
        spike_info: DataFrame
            Spike info for sweep
        end: float
            End of stimulus interval (seconds)
        n_points: int (optional, default 100)
            Number of points in output
        steady_state_interval: float (optional, default 0.1)
            Interval for calculating steady-state for
            sweeps with only one spike (seconds)
        single_return_tolerance: float (optional, default 1)
            Allowable difference from steady-state for
            determining end of "ISI" if only one spike is in sweep (mV)
        single_max_duration: float (optional, default 0.1)
            Allowable max duration for finding end of "ISI"
            if only one spike is in sweep (seconds)

        Returns
        -------
        isi_norm: array of shape (n_points)
            Averaged, threshold-aligned, duration-normalized voltage trace
    """
    
    SF_table=original_SF_table.copy()
    SF_table.index=SF_table.index.astype(int)
    
    TPC_table=original_TPC_table.copy()
    TPC_table.index=TPC_table.index.astype(int)
    SF_table=SF_table.sort_values(by=['Feature','Time_s'])
    n_spikes=SF_table[SF_table['Feature']=='Peak'].shape[0]
    potential_trace=np.array(TPC_table['Membrane_potential_mV'])
    
    if n_spikes > 1:
        threshold_voltages = np.array(SF_table[SF_table["Feature"]=='Threshold'].loc[:,'Membrane_potential_mV'])
        threshold_indexes = np.array(SF_table[SF_table["Feature"]=='Threshold'].index)
        fast_trough_indexes = np.array(SF_table[SF_table["Feature"]=='Fast_Trough'].index)
        
        isi_list = []
        for start_index, end_index, thresh_v in zip(fast_trough_indexes[:-1], threshold_indexes[1:], threshold_voltages[:-1]):
            isi_raw = TPC_table.loc[start_index:end_index,'Membrane_potential_mV'] - thresh_v  #Align to threshold
            width = len(isi_raw) // n_points

            if width == 0:
                logging.debug("found isi shorter than specified width; skipping")
                continue

            isi_norm = _subsample_average(np.array(isi_raw[:width * n_points]), width)
            isi_list.append(isi_norm)

        isi_norm = np.vstack(isi_list).mean(axis=0)
        
    else:
        threshold_v = SF_table[SF_table["Feature"]=='Threshold'].loc[:,'Membrane_potential_mV'].values[0]
        fast_trough_index = SF_table[SF_table["Feature"]=='Fast_Trough'].index[0]
        fast_trough_time = SF_table[SF_table["Feature"]=='Fast_Trough'].loc[:,"Time_s"].values[0]
        fast_trough_t = SF_table[SF_table["Feature"]=='Fast_Trough'].loc[:,'Time_s'].values[0]
        stim_end_index = find_time_index(np.array(TPC_table.loc[:,'Time_s']), end)+TPC_table.index[0]

        if fast_trough_t < end - steady_state_interval:
            
            max_end_index = find_time_index(np.array(TPC_table.loc[:,'Time_s']), fast_trough_time + single_max_duration)+TPC_table.index[0]
            
            std_start_index = find_time_index(np.array(TPC_table.loc[:,'Time_s']), end - steady_state_interval)+TPC_table.index[0]
            
            steady_state_v = TPC_table.loc[std_start_index:stim_end_index,'Membrane_potential_mV'].mean()
            
            above_ss_ind = np.flatnonzero(TPC_table.loc[fast_trough_index:,'Membrane_potential_mV'] >= steady_state_v - single_return_tolerance)+TPC_table.index[0]
            
            if len(above_ss_ind) > 0:
                end_index = above_ss_ind[0] + fast_trough_index
                end_index = min(end_index, max_end_index)
            else:
                logging.debug("isi_shape: voltage does not return to steady-state within specified tolerance; "
                    "resorting to specified max duration")
                end_index = max_end_index
        else:
            end_index = stim_end_index

        # check that it's not too close (less than n_points)
        if end_index - fast_trough_index < n_points:
            if end_index >= stim_end_index:
                logging.warning("isi_shape: spike close to end of stimulus interval")
            end_index = fast_trough_index + n_points
        
        
        isi_raw = np.array(TPC_table.loc[fast_trough_index:end_index,'Membrane_potential_mV'] - threshold_v)
        width = len(isi_raw) // n_points
        isi_raw = isi_raw[:width * n_points] # ensure division will work
        isi_norm = _subsample_average(isi_raw, width)

    return np.array(isi_norm)

def get_isi_norm(sweep_list,original_full_SF_table, original_full_TPC_table,response_duration=.500,do_plot=False):
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()

    
    if len(sweep_list)==1:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[sweep_list[0],'SF'])
        sub_TPC_table=pd.DataFrame(full_TPC_table.loc[sweep_list[0],'TPC'])
        stim_start_time=full_SF_table.loc[sweep_list[0],'Stim_start_s']
        stim_end_time=stim_start_time+response_duration
        isi_norm=isi_shape(original_SF_table=sub_SF_table,
                           original_TPC_table=sub_TPC_table, 
                           end=stim_end_time, 
                           n_points=100, 
                           steady_state_interval=0.1,
                           single_return_tolerance=1.,
                           single_max_duration=0.1)
        
    else:
        full_isi_norm_array=[]
        for current_sweep in sweep_list:

            sub_SF_table=pd.DataFrame(full_SF_table.loc[sweep_list[0],'SF'])
            sub_TPC_table=pd.DataFrame(full_TPC_table.loc[sweep_list[0],'TPC'])
            stim_start_time=full_SF_table.loc[sweep_list[0],'Stim_start_s']
            stim_end_time=stim_start_time+response_duration
            current_isi_norm=isi_shape(original_SF_table=sub_SF_table,
                               original_TPC_table=sub_TPC_table, 
                               end=stim_end_time, 
                               n_points=100, 
                               steady_state_interval=0.1,
                               single_return_tolerance=1.,
                               single_max_duration=0.1)
            full_isi_norm_array.append(current_isi_norm)
        isi_norm=np.vstack(full_isi_norm_array).mean(axis=0)
    
    if do_plot:
        index_table=pd.DataFrame(np.linspace(1,100,100),columns=["Index"])
        isi_table=pd.DataFrame(isi_norm,columns=['delta_mV'])
        table=pd.concat([index_table,isi_table],axis=1)
        my_plot=ggplot(table,aes(x="Index",y='delta_mV'))+geom_line()
        my_plot+=ggtitle("ISI potential shape")
        print(my_plot)
        
    return np.array(isi_norm)
            
def combine_and_interpolate(data):
    """Concatenate and interpolate missing items from neighbors"""

    n_populated = np.sum([d is not None for d in data])
    if n_populated <= 1 and len(data) > 1:
        logging.warning("Data from only one spiking sweep found; interpolated sweeps may have issues")
    output_list = []
    for i, d in enumerate(data):
        if d is not None:
            output_list.append(d)
            continue

        # Missing data is interpolated from neighbors in list
        lower = -1
        upper = np.inf
        for j, neighbor in enumerate(data):
            if j == i: # if current_neighbor is the data currently observed, go to the  next one
                continue
            if neighbor is not None: # if current_neighbor is not empty
                if j < i and j > lower: # get index of the last not empty neighbor before the data currently observed
                    lower = j
                if j > i and j < upper: # get index of the next not empty neighbor after the data currently observed
                    upper = j
        if lower > -1 and upper < len(data): # if the data currently observed is not the forst or last one, interpolate by taking the mean between data[j] and data[i]
            new_data = (data[lower] + data[upper]) / 2
        elif lower == -1: # if the data currently observed is the first in data, interpolating by taking the value of first not empty data
            new_data = data[upper]
        else:
            new_data = data[lower]

        output_list.append(new_data) # if the data currently observed is the last in data, interpolating by taking the value of last not empty data

    return np.hstack(output_list)


def identify_sweep_for_psth(original_full_SF_table, original_full_TPC_table, original_metadata_table,stim_amp_step=20, stim_amp_step_margin=5):
    """ Find lowest-amplitude spiking sweep that has at least min_spike
        or else sweep with most spikes

        Parameters
        ----------
        sweeps: SweepSet
            Sweeps to consider for ISI shape calculation
        features: dict
            Output of LongSquareAnalysis.analyze()
        duration: float
            Length of stimulus interval (seconds)
        min_spike: int (optional, default 5)
            Minimum number of spikes for first preference sweep (default 5)

        Returns
        -------
        selected_sweep: Sweep
            Sweep object for ISI shape calculation
        selected_spike_info: DataFrame
            Spike info for selected sweep
    """
    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike
    
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()
    metadata_table=original_metadata_table.copy()
    
    sweep_list=np.array(full_SF_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocols'].values[0])
    Nb_of_Traces=full_SF_table.shape[0]

    
    selected_sweeps_psth=[]
    identified_rheobase=False
    end_rheobase=0
    previous_stim_amp_avg=0
    for elt in range(Nb_of_Traces//Nb_of_Protocols):
        current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
        current_max_nb_of_spike=0
        for current_idx in current_sweeps_idx:
            current_sweep=sweep_list[current_idx]
            current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
            current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
        current_stim_amp_avg=np.mean(np.array(full_SF_table.loc[sweep_list[current_sweeps_idx],:]['Stim_amp_pA']))
        
        if current_max_nb_of_spike>=1 and identified_rheobase == False:
            identified_rheobase=True
            rheobase=current_stim_amp_avg
            selected_sweeps_psth.append(np.array(sweep_list[current_sweeps_idx]))
            end_rheobase=rheobase+100
            previous_stim_amp_avg=rheobase
        
        elif current_stim_amp_avg <= end_rheobase and identified_rheobase==True:
            #if in the protocol, the stimulus step is higher than 20pA(stim_amp_step), 
            #add false sweep id, to interpolate in the next function
            #add a 5pA error margin to avoid false 20pA-steps i.e.: (-24.344--64.643)//20
            step_amp=int((current_stim_amp_avg-previous_stim_amp_avg)//(stim_amp_step+stim_amp_step_margin)) 
            
            for current_step in range(step_amp): # if step_amp==0 (so no 20pA step skipped), the loop is null, so nothing is added
                selected_sweeps_psth.append(np.zeros_like(current_sweeps_idx))
            selected_sweeps_psth.append(np.array(sweep_list[current_sweeps_idx]))
            previous_stim_amp_avg=current_stim_amp_avg
            
        elif current_stim_amp_avg > end_rheobase and identified_rheobase==True:
            break
           
    return np.array(selected_sweeps_psth)

def psth_vector(selected_sweeps_psth,original_full_SF_table, width=50,response_duration=.500,do_plot=False):
    """ Create binned "PSTH"-like feature vector based on spike times, concatenated
        across sweeps

    Parameters
    ----------
    spike_info_list: list
        Spike info DataFrames for each sweep
    start: float
        Start of stimulus interval (seconds)
    end: float
        End of stimulus interval (seconds)
    width: float (optional, default 50)
        Bin width in ms
    response_duration : float (optional, default .5)
        Duration of the response to take into account

    Returns
    -------
    output: array
        Concatenated vector of binned spike rates (spikes/s)
    """
    SF_table=original_full_SF_table.copy()
    vector_list = []

    for current_sweep_list in selected_sweeps_psth:
        output=[]
        if current_sweep_list[0] == 0:
            vector_list.append(None)
            continue
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=SF_table.loc[current_sweep,'SF']
            current_stim_start=SF_table.loc[current_sweep,'Stim_start_s']
            current_stim_end=current_stim_start+response_duration
            
            current_spike_times=np.array(current_SF_table[current_SF_table['Feature']=='Upstroke']['Time_s'])
            current_spike_times=current_spike_times[current_spike_times<current_stim_end]
            spike_count = np.ones_like(current_spike_times)
            one_ms = 0.001
            duration = np.round(current_stim_end, decimals=3) - np.round(current_stim_start, decimals=3)
            n_bins = int(duration / one_ms) // width
            bin_edges = np.linspace(current_stim_start, current_stim_end, n_bins + 1) # includes right edge, so adding one to desired bin number
            bin_width = bin_edges[1] - bin_edges[0]
            current_output,current_bin_edges,current_bins_number = scipy.stats.binned_statistic(current_spike_times,
                                            spike_count,
                                            statistic='sum',
                                            bins=bin_edges)
           

            current_output[np.isnan(current_output)] = 0
            current_output /= bin_width # convert to spikes/s
            output.append(current_output)
        output = np.vstack(output).mean(axis=0)
        vector_list.append(output)

        
    output_vector = combine_and_interpolate(vector_list)
    if do_plot:
        plt.hlines(output_vector,np.arange(0,len(output_vector)+1)[:-1],np.arange(0,len(output_vector)+1)[1:],label='50ms_bin')
        plt.vlines(np.arange(0,len(output_vector)+1,10),ymin=0,ymax=60,linestyles='dashed',color='red',label='500ms_sweep_limit')
        plt.ylabel('Spikes/s')
        plt.legend(fontsize=10,loc='lower right')
        plt.xlabel('Bin_index')
        
    return output_vector

def _inst_freq_feature(spike_times, start, end):
    """ Estimate instantaneous firing rate from differences in spike times

    This function attempts to estimate a semi-continuous instantanteous firing
    rate from a set of interspike intervals (ISIs) and spike times. It makes
    several assumptions:
    - It assumes that the instantaneous firing rate at the start of the stimulus
    interval is the inverse of the latency to the first spike.
    - It estimates the firing rate at each spike as the average of the ISIs
    on each side of the spike
    - If the time between the end of the interval and the last spike is less
    than the last true interspike interval, it sets the instantaneous rate of
    that last spike and of the end of the interval to the inverse of the last ISI.
    Therefore, the instantaneous rate would not "jump" just because the
    stimulus interval ends.
    - However, if the time between the end of the interval and the last spike is
    longer than the final ISI, it assumes there may have been a spike just
    after the end of the interval. Therefore, it essentially returns an upper
    bound on the estimated rate.


    Parameters
    ----------
    threshold_t: array
        Spike times
    start: float
        start of stimulus interval (seconds)
    end: float
        end of stimulus interval (seconds)

    Returns
    -------
    inst_firing_rate: array of shape (len(threshold_t) + 2)
        Instantaneous firing rate estimates (spikes/s)
    time_points: array of shape (len(threshold_t) + 2)
        Time points for corresponding values in inst_firing_rate
    """
    if len(spike_times) == 0:
        return np.array([0, 0]), np.array([start, end])
    inst_inv_rate = []
    time_points = []
    isis = [(spike_times[0] - start)] + np.diff(spike_times).tolist()
    isis = isis + [max(isis[-1], end - spike_times[-1])]

    # Estimate at start of stimulus interval
    inst_inv_rate.append(isis[0])
    time_points.append(start)

    # Estimate for each spike time
    for t, pre_isi, post_isi in zip(spike_times, isis[:-1], isis[1:]):
        inst_inv_rate.append((pre_isi + post_isi) / 2)
        time_points.append(t)

    # Estimate for end of stimulus interval
    inst_inv_rate.append(isis[-1])
    time_points.append(end)


    inst_firing_rate = 1 / np.array(inst_inv_rate)
    time_points = np.array(time_points)
    return inst_firing_rate, time_points

def inst_freq_vector(selected_sweeps_inst_freq,original_full_SF_table, width=20,response_duration=.500,normalized=False,do_plot=False):
    """ Create binned instantaneous frequency feature vector,
        concatenated across sweeps

    Parameters
    ----------
    spike_info_list: list
        Spike info DataFrames for each sweep
    start: float
        Start of stimulus interval (seconds)
    end: float
        End of stimulus interval (seconds)
    width: float (optional, default 20)
        Bin width in ms


    Returns
    -------
    output: array
        Concatenated vector of binned instantaneous firing rates (spikes/s)
    """
    SF_table=original_full_SF_table.copy()
    vector_list = []
    for current_sweep_list in selected_sweeps_inst_freq:
        output=[]
        if current_sweep_list[0] == 0:
            vector_list.append(None)
            continue
        
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=SF_table.loc[current_sweep,'SF']
            current_stim_start=SF_table.loc[current_sweep,'Stim_start_s']
            current_stim_end=current_stim_start+response_duration
            
            current_spike_times=np.array(current_SF_table[current_SF_table['Feature']=='Upstroke']['Time_s'])
            current_spike_times=current_spike_times[current_spike_times<current_stim_end]
            inst_freq, inst_freq_times = _inst_freq_feature(current_spike_times, current_stim_start, current_stim_end)
            one_ms = 0.001

            # round to nearest ms to deal with float approximations
            duration = np.round(current_stim_end, decimals=3) - np.round(current_stim_start, decimals=3)

            n_bins = int(duration / one_ms) // width
            bin_edges = np.linspace(current_stim_start, current_stim_end, n_bins + 1) # includes right edge, so adding one to desired bin number
            bin_width = bin_edges[1] - bin_edges[0]
            current_output = scipy.stats.binned_statistic(inst_freq_times,
                                            inst_freq,
                                            bins=bin_edges)[0]
            nan_ind = np.isnan(current_output)
            x = np.arange(len(current_output))
            current_output[nan_ind] = np.interp(x[nan_ind], x[~nan_ind], current_output[~nan_ind])
            if normalized == True:
                current_output/=max(current_output)
            output.append(current_output)
        
        output = np.vstack(output).mean(axis=0)
        vector_list.append(output)

        
    output_vector = combine_and_interpolate(vector_list)
            
    if do_plot:
        plt.hlines(output_vector,np.arange(0,len(output_vector)+1)[:-1],np.arange(0,len(output_vector)+1)[1:],label='20ms_bin')
        plt.vlines(np.arange(0,len(output_vector)+1,25),ymin=0,ymax=max(output_vector),linestyles='dashed',color='red',label='500ms_sweep_limit')
        plt.ylabel('Inst_freq_Hz')
        plt.legend(fontsize=10,loc='lower right')
        plt.xlabel('Bin_index')

    return output_vector



def find_widths(original_SF_table,original_TPC_table, clipped=None):
    """Find widths at half-height for spikes.

    Widths are only returned when heights are defined

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike threshold indexes
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of trough indexes (trough= minimum potential value between a peak and the next threshold)

    Returns
    -------
    widths : numpy array of spike widths in sec
    """
    TPC_table=pd.DataFrame(original_TPC_table).copy()
    SF_table=pd.DataFrame(original_SF_table).copy()
    
    threshold_table=SF_table[SF_table['Feature']=='Threshold'].sort_values(by=['Time_s']).copy()
    threshold_potential_array=np.array(threshold_table['Membrane_potential_mV'])
    threshold_time_array=np.array(threshold_table['Time_s'])
    threshold_index=np.array(threshold_table.index,dtype=int)-int(TPC_table.index[0])
    
    peak_table=SF_table[SF_table['Feature']=='Peak'].sort_values(by=['Time_s']).copy()
    peak_potential_array=np.array(peak_table['Membrane_potential_mV'])
    peak_time_array=np.array(peak_table['Time_s'])
    peak_index=np.array(peak_table.index,dtype=int)-int(TPC_table.index[0])
    
    trough_table=SF_table[SF_table['Feature']=='Trough'].sort_values(by=['Time_s']).copy()
    trough_potential_array=np.array(trough_table['Membrane_potential_mV'])
    trough_time_array=np.array(trough_table['Time_s'])
    trough_index=np.array(trough_table.index,dtype=int)-int(TPC_table.index[0])
    
    potential_trace=np.array(TPC_table['Membrane_potential_mV'])
    time_array=np.array(TPC_table['Time_s'])
    
    
    
    
    if not threshold_time_array.size or not peak_time_array.size:
        return np.array([])
    
    if len(peak_time_array)==(len(trough_time_array)+1):

        peak_time_array=peak_time_array[:-1]
        peak_potential_array=peak_potential_array[:-1]
        peak_index=peak_index[:-1]
        
        threshold_potential_array=threshold_potential_array[:-1]
        threshold_time_array=threshold_time_array[:-1]
        threshold_index=threshold_index[:-1]
    # if len(threshold_time_array) < len(trough_time_array):
    #     raise er.FeatureError("Cannot have more troughs than spikes")

    if clipped is None:
        clipped = np.zeros_like(peak_time_array, dtype=bool)

    
    use_indexes = ~np.isnan(trough_time_array)
    
    use_indexes[clipped] = False

    heights = np.zeros_like(trough_time_array) * np.nan
    heights[use_indexes] = peak_potential_array[use_indexes] - trough_potential_array[use_indexes]
   # heights[use_indexes] = v[peak_indexes[use_indexes]] - v[trough_indexes[use_indexes].astype(int)]

    width_levels = np.zeros_like(trough_time_array) * np.nan
    width_levels[use_indexes] = heights[use_indexes] / 2. + trough_potential_array[use_indexes]
    #width_levels[use_indexes] = heights[use_indexes] / 2. + v[trough_indexes[use_indexes].astype(int)]


    thresh_to_peak_levels = np.zeros_like(trough_time_array) * np.nan
    thresh_to_peak_levels[use_indexes] = (peak_potential_array[use_indexes] - threshold_potential_array[use_indexes]) / 2. + threshold_potential_array[use_indexes]
    #thresh_to_peak_levels[use_indexes] = (v[peak_indexes[use_indexes]] - v[spike_indexes[use_indexes]]) / 2. + v[spike_indexes[use_indexes]]

    # Some spikes in burst may have deep trough but short height, so can't use same
    # definition for width

    width_levels[width_levels < threshold_potential_array] = thresh_to_peak_levels[width_levels < threshold_potential_array]

    width_starts = np.zeros_like(trough_time_array) * np.nan
    width_starts[use_indexes] = np.array([pk - np.flatnonzero(potential_trace[pk:spk:-1] <= wl)[0] if
                    np.flatnonzero(potential_trace[pk:spk:-1] <= wl).size > 0 else np.nan for pk, spk, wl
                    in zip(peak_index[use_indexes], threshold_index[use_indexes], width_levels[use_indexes])])
    
    # width_starts[use_indexes] = np.array([pk - np.flatnonzero(v[pk:spk:-1] <= wl)[0] if
    #                 np.flatnonzero(v[pk:spk:-1] <= wl).size > 0 else np.nan for pk, spk, wl
    #                 in zip(peak_indexes[use_indexes], spike_indexes[use_indexes], width_levels[use_indexes])])
    width_ends = np.zeros_like(trough_time_array) * np.nan

    width_ends[use_indexes] = np.array([pk + np.flatnonzero(potential_trace[pk:tr] <= wl)[0] if
                    np.flatnonzero(potential_trace[pk:tr] <= wl).size > 0 else np.nan for pk, tr, wl
                    in zip(peak_index[use_indexes], trough_index[use_indexes].astype(int), width_levels[use_indexes])])
    # width_ends[use_indexes] = np.array([pk + np.flatnonzero(v[pk:tr] <= wl)[0] if
    #                 np.flatnonzero(v[pk:tr] <= wl).size > 0 else np.nan for pk, tr, wl
    #                 in zip(peak_indexes[use_indexes], trough_indexes[use_indexes].astype(int), width_levels[use_indexes])])

    missing_widths = np.isnan(width_starts) | np.isnan(width_ends)
    widths = np.zeros_like(width_starts, dtype=np.float64)
    widths[~missing_widths] = time_array[width_ends[~missing_widths].astype(int)] - \
                              time_array[width_starts[~missing_widths].astype(int)]
    if any(missing_widths):
        widths[missing_widths] = np.nan

    return widths




def binned_AP_single_feature(selected_sweeps_inst_freq,original_full_SF_table, original_full_TPC_table, feature='Peak',width=20,response_duration=.500,do_plot=False):
    """ Create binned instantaneous frequency feature vector,
        concatenated across sweeps

    Parameters
    ----------
    spike_info_list: list
        Spike info DataFrames for each sweep
    start: float
        Start of stimulus interval (seconds)
    end: float
        End of stimulus interval (seconds)
    width: float (optional, default 20)
        Bin width in ms


    Returns
    -------
    output: array
        Concatenated vector of binned instantaneous firing rates (spikes/s)
    """
    SF_table=original_full_SF_table.copy()
    TPC_table=original_full_TPC_table.copy()
    vector_list = []
    Original_data_vector=[]
    n_bins=int(response_duration*1e3//width)
    for current_sweep_list in selected_sweeps_inst_freq:
        output=[]
        sweep_col=[]
        bin_columns=[]
        Stim_table=pd.DataFrame(columns=['Bin_index',str(feature),'Original_Data','Sweep'])
        if current_sweep_list[0] == 0:
            vector_list.append(None)
            Original_data_vector.append(np.ones(n_bins,dtype=bool))
            continue
        
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=pd.DataFrame(SF_table.loc[current_sweep,'SF'].copy())
            current_stim_start=SF_table.loc[current_sweep,'Stim_start_s']
            current_stim_end=current_stim_start+response_duration
            if feature=='Up/Down':
                current_Up_table=current_SF_table[current_SF_table['Feature']=='Upstroke']
                current_Down_table=current_SF_table[current_SF_table['Feature']=='Downstroke']
                
                current_Up_time=np.array(current_Up_table['Time_s'])
                current_Up_potential=np.array(current_Up_table['Membrane_potential_mV'])
                
                current_Down_time=np.array(current_Down_table['Time_s'])
                current_Down_potential=np.array(current_Down_table['Membrane_potential_mV'])
                
                if current_Up_table.shape[0]==current_Down_table.shape[0]+1:
                    current_Up_time=current_Up_time[:-1]
                    current_Up_potential=current_Up_potential[:-1]
                
                
                Up_Down_table=pd.DataFrame()
                Up_Down_table['Time_s']=pd.Series(np.array((current_Up_time+current_Down_time)/2))
                Up_Down_table['Membrane_potential_mV']=pd.Series(np.array(current_Up_potential/current_Down_potential))
                
                
                current_feature_time=np.array(Up_Down_table[Up_Down_table['Time_s']<=current_stim_end]['Time_s'])
                current_feature_value=np.array(Up_Down_table[Up_Down_table['Time_s']<=current_stim_end]['Membrane_potential_mV'])
                
            elif feature=='Width':
                current_TPC_table=TPC_table.loc[current_sweep,'TPC'].copy()
                trough_table=current_SF_table[current_SF_table['Feature']=='Trough']
                trough_time=np.array(trough_table['Time_s'])
                current_feature_value = find_widths(current_SF_table, current_TPC_table)
                current_feature_time=trough_time
                #width_array = find_widths(v, t, spike_indexes, peak_indexes, trough_indexes)
            else:
                current_table=current_SF_table[current_SF_table['Time_s']<=current_stim_end]
                current_feature_time=np.array(current_table[current_table['Feature']==feature]['Time_s'])
                current_feature_value=np.array(current_table[current_table['Feature']==feature]['Membrane_potential_mV'])
            
            one_ms = 0.001

            # round to nearest ms to deal with float approximations
            duration = np.round(current_stim_end, decimals=3) - np.round(current_stim_start, decimals=3)


            bin_edges = np.linspace(current_stim_start, current_stim_end, n_bins + 1)
            print('bin_edge',bin_edges)# includes right edge, so adding one to desired bin number
            bin_width = bin_edges[1] - bin_edges[0]
            current_output = scipy.stats.binned_statistic(current_feature_time,
                                            current_feature_value,
                                            statistic='mean',
                                            bins=bin_edges)[0]

            nan_ind = np.isnan(current_output) #identify bins where there is no value
            not_nan_ind = ~np.isnan(current_output) 

            x = np.arange(len(current_output))
            current_output[nan_ind] = np.interp(x[nan_ind], x[~nan_ind], current_output[~nan_ind]) #if bin has no value, interpolate by taking the mean between the last and next not-nan value
            #if more than one null value in a row, linear estimation (one third, then two third of the diff...)
            if do_plot==True:
                table=pd.DataFrame()
                table['Bin_index']=pd.Series(np.arange(0,len(current_output)))
                table[str(feature)]=pd.Series(current_output)

                table['Original_Data']=not_nan_ind
                table['Sweep']=str(current_sweep)
                my_plot=ggplot()+geom_point(table,aes(x='Bin_index',y=table.iloc[:,1],shape='Original_Data'))
                my_plot+=ggtitle(str('Sweep:'+str(current_sweep)))
                print(my_plot)
            Stim_table=pd.concat([Stim_table,table],axis=0)
            output.append(current_output)
            

        original_output=output
        
        
        Original_data_vector.append(np.zeros(n_bins,dtype=bool))
        sweep_col.append(np.array(['Mean_value']*n_bins))
        output = np.vstack(output).mean(axis=0)
        bin_columns.append(np.arange(0,len(current_output)))
        
        vector_list.append(output)
        original_output.append(output)
        
        if do_plot:
            table=pd.DataFrame()
            table['Bin_index']=pd.Series(np.arange(0,len(current_output)))
            table[str(feature)]=pd.Series(output)
            table['Original_Data']=False
            table['Sweep']=str("Mean_value")
            Stim_table=pd.concat([Stim_table,table],axis=0)
            
            stim_plot=ggplot(Stim_table,aes(x="Bin_index",y=Stim_table.iloc[:,1],color='Sweep'))+geom_point()+ggtitle(str(str(feature)+"_averaged_over_sweeps_"+str(current_sweep_list)))
            print(stim_plot)
        
        
    output_vector = combine_and_interpolate(vector_list)

    if do_plot:

        Original_data_vector=np.hstack(Original_data_vector)
        Full_table=pd.DataFrame()
        Full_table['Bin_index']=pd.Series(np.arange(len(output_vector)))
        Full_table[str(feature)]=pd.Series(output_vector)
        Full_table['Extrapolated_sweep']=pd.Series(Original_data_vector)
        my_plot=ggplot(Full_table,aes(x='Bin_index',y=Full_table.iloc[:,1],color='Extrapolated_sweep'))+geom_point()
        my_plot+=geom_vline(xintercept=(np.arange(len(selected_sweeps_inst_freq))*n_bins)-1)
        print(Full_table)
        
        if feature=='Up/Down':

            my_plot+=ggtitle("Upstroke/Downstroke_ratio")
        elif feature=='Width':
            my_plot+=ggtitle('AP_width_s')
        else:
            my_plot+=ggtitle(str(feature+'_potential_mV'))
            
        print(my_plot)
    return output_vector


def identify_subthreshold_hyperpol_with_amplitudes(original_full_TPC_table,original_full_SF_table, original_Metadata_table):
    """ Identify subthreshold responses from hyperpolarizing steps

        Parameters
        ----------
        features: dict
            Output of LongSquareAnalysis.analyze()
        sweeps: SweepSet
            Long square sweeps

        Returns
        -------
        amp_sweep_dict: dict
            Amplitude-sweep pairs
        deflect_dict: dict
            Dictionary of (base, deflect) tuples with amplitudes as keys
    """
    
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()
    Metadata_table=original_Metadata_table.copy()
    
    sweep_list=np.array(full_SF_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(Metadata_table.loc[:,'Nb_of_Protocols'].values[0])
    Nb_of_Traces=full_SF_table.shape[0]

    
    sub_threshold_sweep=[]
    sub_threshold_hyperpol_sweeps=[]

    for elt in range(Nb_of_Traces//Nb_of_Protocols):
        current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
        current_max_nb_of_spike=0
        for current_idx in current_sweeps_idx:
            current_sweep=sweep_list[current_idx]
            current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
            current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
        if current_max_nb_of_spike == 0:
            sub_threshold_sweep.append(list(sweep_list[current_sweeps_idx]))
            
    

            
    for current_subthresh_sweeps_list in sub_threshold_sweep:
        depolarizing_sweep=0
        
        for current_subthresh_sweeps in current_subthresh_sweeps_list:
            current_TPC=pd.DataFrame(full_TPC_table.loc[current_subthresh_sweeps,'TPC'])
            current_stim_start=full_SF_table.loc[current_subthresh_sweeps,'Stim_start_s']
            
            potential_baseline=np.mean(np.array(current_TPC[current_TPC['Time_s']<(current_stim_start-.005)]['Membrane_potential_mV']))
            after_stim_potential_table=current_TPC[current_TPC['Time_s']>(current_stim_start+.005)]
            after_stim_potential_table=after_stim_potential_table[after_stim_potential_table['Time_s']<current_stim_start+.5]
            potential_after_stim_start=np.mean(np.array(after_stim_potential_table['Membrane_potential_mV']))
           
            if potential_baseline < potential_after_stim_start: #if current_sweep is depolarizing
                depolarizing_sweep+=1
                
        if depolarizing_sweep == 0: # if no sweep in the sublist (in case of duplicate/triplicate) is depolarizing --> hyperpolarizing

            sub_threshold_hyperpol_sweeps.append(current_subthresh_sweeps_list)
            
    
            
     
    return np.array(sub_threshold_hyperpol_sweeps)

def bisidentify_subthreshold_hyperpol_with_amplitudes(features, sweeps):
    """ Identify subthreshold responses from hyperpolarizing steps

        Parameters
        ----------
        features: dict
            Output of LongSquareAnalysis.analyze()
        sweeps: SweepSet
            Long square sweeps

        Returns
        -------
        amp_sweep_dict: dict
            Amplitude-sweep pairs
        deflect_dict: dict
            Dictionary of (base, deflect) tuples with amplitudes as keys
    """

    # Get non-spiking sweeps
    subthresh_df = features["subthreshold_sweeps"]

    # Get responses to hyperpolarizing steps
    subthresh_df = subthresh_df.loc[subthresh_df["stim_amp"] < 0] # only consider hyperpolarizing steps

    # Construct dictionary
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweeps.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)
    subthresh_deflect = subthresh_df["peak_deflect"].values
    subthresh_base = subthresh_df["v_baseline"].values

    mask = subthresh_amps < -1000 # TEMP QC ISSUE: Check for bad amps; shouldn't have to do this in general
    subthresh_amps = subthresh_amps[~mask]
    subthresh_sweeps = subthresh_sweeps[~mask]
    subthresh_deflect = subthresh_deflect[~mask]
    subthresh_base = subthresh_base[~mask]

    amp_sweep_dict = dict(zip(subthresh_amps, subthresh_sweeps))
    base_deflect_tuples = zip(subthresh_base, [d[0] for d in subthresh_deflect])
    deflect_dict = dict(zip(subthresh_amps, base_deflect_tuples))
    return amp_sweep_dict, deflect_dict

def step_subthreshold(selected_sweep_list,original_full_TPC_table,original_full_SF_table, target_amps,
                      extend_duration=0.2, subsample_interval=0.01,
                      amp_tolerance=0.):
    """ Subsample set of subthreshold step responses including regions before and after step

        Parameters
        ----------
        amp_sweep_dict : dict
            Amplitude-sweep pairs
        target_amps: list
            Desired amplitudes for output vector
        start: float
            start stimulus interval (seconds)
        end: float
            end of stimulus interval (seconds)
        extend_duration: float (optional, default 0.2)
            Duration to extend sweep before and after stimulus interval (seconds)
        subsample_interval: float (optional, default 0.01)
            Size of subsampled bins (seconds)
        amp_tolerance: float (optional, default 0)
            Tolerance for finding matching amplitudes

        Returns
        -------
        output_vector: subsampled, concatenated voltage trace
    """

    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()
    
    extend_length = int(np.round(extend_duration / subsample_interval))
    subsampled_dict = {}
    
    for current_sweep_sub_list in selected_sweep_list:
        mean_amp=np.mean(full_SF_table.loc[current_sweep_sub_list,'Stim_amp_pA'])
        potential_list=[]
        for current_sweep in current_sweep_sub_list:
            
            current_TPC=pd.DataFrame(full_TPC_table.loc[current_sweep,'TPC'])
            current_TPC=current_TPC.reset_index(drop=True)
            

            current_time_array=np.array(current_TPC['Time_s'])
            current_potential_array = np.array(current_TPC['Membrane_potential_mV'])
            
            current_stim_start=full_SF_table.loc[current_sweep,'Stim_start_s']
            current_start_index=find_time_index(current_time_array, current_stim_start - extend_duration)
            
            current_stim_end=full_SF_table.loc[current_sweep,'Stim_end_s']
            current_end_index=find_time_index(current_time_array, current_stim_end + extend_duration)

            delta_t = current_time_array[1] - current_time_array[0]
            subsample_width = int(np.round(subsample_interval / delta_t))
            current_subsampled_potential = _subsample_average(current_potential_array[current_start_index:current_end_index], subsample_width)
            potential_list.append(current_subsampled_potential)
            
            
        subsampled_potential = np.vstack(potential_list).mean(axis=0)
        subsampled_dict[mean_amp] = subsampled_potential
            
    # Subsample each sweep
   
    
    available_amps = np.array(list(subsampled_dict.keys()))
    output_list = []
    for amp in target_amps: # take one amplitude in targets
        amp_diffs = np.array(np.abs(available_amps - amp)) # difference between current all amplitudes and current traget amplitude

        if np.any(amp_diffs <= amp_tolerance): # if one sweep amplitude matches current target amplitude with tolerance
            matching_amp = available_amps[np.argmin(amp_diffs)] #take matching amplitude
            logging.debug("found amp of {} to match {} (tol={})".format(matching_amp, amp, amp_tolerance))
            output_list.append(subsampled_dict[matching_amp]) # add corresponding potential trace to list 
            
        else: # if no sweep amplitude matches current target amplitude with tolerance
            lower_amp = False
            upper_amp = False
            for a in available_amps: # test all available amplitudes
                if a < amp: # if current available amplitude is lower than  current target amplitude, find closest lower available amp
                    if lower_amp == False:
                        lower_amp = a
                    elif a > lower_amp: #if current available amplitude is higher than lastly defined lowest amp replace it
                        lower_amp = a
                if a > amp:  #if current available amplitude is higher than current target amplitude , find closest higher available amp
                    if upper_amp == False:
                        upper_amp = a
                    elif a < upper_amp:
                        upper_amp = a
            if lower_amp != False and upper_amp != False:
                logging.debug("interpolating for amp {} with lower {} and upper {}".format(amp, lower_amp, upper_amp))
                avg = (subsampled_dict[lower_amp] + subsampled_dict[upper_amp]) / 2.
                scale = amp / ((lower_amp + upper_amp) / 2.)
                base_v = avg[:extend_length].mean() #mean potential value before stim start
                avg[extend_length:-extend_length] = (avg[extend_length:-extend_length] - base_v) * scale + base_v
            elif lower_amp != False:
                logging.debug("interpolating for amp {} from lower {}".format(amp, lower_amp))
                avg = subsampled_dict[lower_amp].copy()
                scale = amp / lower_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            elif upper_amp != False:
                logging.debug("interpolating for amp {} from upper {}".format(amp, upper_amp))
                avg = subsampled_dict[upper_amp].copy()
                scale = amp / upper_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            output_list.append(avg)
            
            
    
    return np.hstack(output_list)


    
def subthresh_norm(selected_sweep_list,original_full_TPC_table,original_full_SF_table ,#Need to adapt to get minimum current amplitude
                   extend_duration=0.2, subsample_interval=0.01):
    """ Subthreshold step response closest to target amplitude normalized to baseline and peak deflection

        Parameters
        ----------
        amp_sweep_dict: dict
            Amplitude-sweep pairs
        deflect_dict:
            Dictionary of (baseline, deflect) tuples with amplitude keys
        start: float
            start stimulus interval (seconds)
        end: float
            end of stimulus interval (seconds)
        target_amp: float (optional, default=-101)
            Search target for amplitude (pA)
        extend_duration: float (optional, default 0.2)
            Duration to extend sweep on each side of stimulus interval (seconds)
        subsample_interval: float (optional, default 0.01)
            Size of subsampled bins (seconds)

        Returns
        -------
        subsampled_v: array
            Subsampled, normalized voltage trace
    """
    
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()
    
    extend_length = int(np.round(extend_duration / subsample_interval))

    minimum_amplitude=np.inf
    for current_sweep_sub_list in selected_sweep_list:
        current_mean_amp=np.mean(full_SF_table.loc[current_sweep_sub_list,'Stim_amp_pA'])

        if current_mean_amp<minimum_amplitude:
            minimum_amplitude=current_mean_amp

            potential_list=[]
            for current_sweep in current_sweep_sub_list:
                
                current_TPC=pd.DataFrame(full_TPC_table.loc[current_sweep,'TPC'])
                current_TPC=current_TPC.reset_index(drop=True)
                
    
                current_time_array=np.array(current_TPC['Time_s'])
                current_potential_array = np.array(current_TPC['Membrane_potential_mV'])
                
                current_stim_start=full_SF_table.loc[current_sweep,'Stim_start_s']
                current_start_index=find_time_index(current_time_array, current_stim_start - extend_duration)
                
                current_stim_end=full_SF_table.loc[current_sweep,'Stim_end_s']
                current_end_index=find_time_index(current_time_array, current_stim_end + extend_duration)
    
                delta_t = current_time_array[1] - current_time_array[0]
                subsample_width = int(np.round(subsample_interval / delta_t))
                current_subsampled_potential = _subsample_average(current_potential_array[current_start_index:current_end_index], subsample_width)
                
                
                potential_list.append(current_subsampled_potential)
                
                
            minimum_amp_potential = np.vstack(potential_list).mean(axis=0)
            base_v = minimum_amp_potential[:extend_length].mean()
            peak_deflect=minimum_amp_potential[extend_length:-extend_length].min()
            delta=base_v-peak_deflect
            
            normalized_minimum_amp_potential = minimum_amp_potential-base_v
            normalized_minimum_amp_potential /= delta
            plt.plot(np.array(normalized_minimum_amp_potential))
    return normalized_minimum_amp_potential
    
            
            
        

    # sweep_ind = np.argmin(np.abs(available_amps - target_amp))
    # matching_amp = available_amps[sweep_ind]
    # swp = amp_sweep_dict[matching_amp]
    # base, deflect_v = deflect_dict[matching_amp]
    # delta = base - deflect_v

    # start_index = tsu.find_time_index(swp.t, start - extend_duration)
    # delta_t = swp.t[1] - swp.t[0]
    # subsample_width = int(np.round(subsample_interval / delta_t))
    # end_index = tsu.find_time_index(swp.t, end + extend_duration)
    # subsampled_v = _subsample_average(swp.v[start_index:end_index], subsample_width)
    # subsampled_v -= base
    # subsampled_v /= delta

    # return subsampled_v
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    