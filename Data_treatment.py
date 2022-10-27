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
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram
import scipy
from scipy.stats import linregress
from scipy import optimize
from scipy.optimize import curve_fit
import random
from tqdm import tqdm


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

#%%
ctc= CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
#%%
def data_pruning (stim_freq_table):
    
    stim_freq_table.sort_values(by=["Cell_id", 'Stim_amp_pA','Frequency_Hz'])
    
    frequency_array=np.array(stim_freq_table.loc[:,'Frequency_Hz'])
    
    
    step_array=np.diff(frequency_array)
    if np.count_nonzero(frequency_array)<4:
        obs='Less_than_4_response'
        do_fit=False
        return obs,do_fit
    if np.count_nonzero(step_array)<3 :
        obs='Less_than_3_different_frequencies'
        do_fit=False
        return obs,do_fit
        
    first_non_zero=np.flatnonzero(frequency_array)[0]
    stim_array=np.array(stim_freq_table.loc[:,'Stim_amp_pA'])[first_non_zero:]
    stimulus_span=stim_array[-1]-stim_array[0]
    if stimulus_span<100.:
        obs='Stimulus_span_lower_than_100pA'
        return obs,do_fit
    
    count,bins=np.histogram(stim_array,
         bins=int((stim_array[-1]+(10 - stimulus_span % 10)-stim_array[0])/10),
          range=(stim_array[0], stim_array[-1]+(10 - stimulus_span % 10)))
          
        
    different_stim=len(np.flatnonzero(count))
    
    if different_stim <5:
        obs='Less_than_5_different_stim_amp'
        do_fit=False
        return obs,do_fit
    
    obs='-'
    do_fit=True
    return obs,do_fit

def extract_stim_freq(Cell_id,species_sweep_stim_table,per_time=False,first_x_ms=0,per_nth_spike=False,first_nth_spike=0):
    '''
    Function to extract for each specified Cell_id and the corresponding stimulus the frequency of the response
    Frequency is defined as the number of spikes divided by the time between the stimulus start and the time of the specified index
    Parameters
    ----------
    Cell_id : int

    
    Returns
    -------
    f_I_table : DataFrame
        DataFrame with a column "Cell_id"(factor),the sweep number (int),the stimulus amplitude in pA(float),and the computed frequency of the response (float).
    '''
    
    f_I_table = pd.DataFrame(columns=['Cell_id', 'Sweep_number', 'Stim_amp_pA', 'Frequency_Hz'])
    
    my_Cell_data = ctc.get_ephys_data(Cell_id)
    
    cell_stim_spike_table=extract_stim_spike_times_table(Cell_id,mouse_sweep_stim_table,do_save=False)

    for current_line in range(cell_stim_spike_table.shape[0]):
        current_sweep=cell_stim_spike_table.iloc[current_line,1]

        
        stim_start_time=cell_stim_spike_table.iloc[current_line,3]
        spike_times=cell_stim_spike_table.iloc[current_line,5]
        
        
        if len(spike_times) <1:
            freq = 0

        else :
            if per_nth_spike==True:
                reshaped_spike_times=spike_times[:first_nth_spike]
    
                t_last_spike = reshaped_spike_times[-1]

                freq=len(reshaped_spike_times)/((t_last_spike - stim_start_time))

            elif per_time==True:
                end_time=stim_start_time+(first_x_ms*1e-3)

                reshaped_spike_times=spike_times[spike_times <= end_time ]
                nb_spike = len(reshaped_spike_times)

                if nb_spike !=0:
                    
                    freq=nb_spike/(first_x_ms*1e-3)
                else:
                    freq=0
        new_line = pd.Series([int(Cell_id), current_sweep,
                              my_Cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa'],
                              freq],
                             index=['Cell_id', 'Sweep_number', 'Stim_amp_pA', 'Frequency_Hz'])
        f_I_table = f_I_table.append(new_line, ignore_index=True)
    
    freq_array=f_I_table.iloc[:,3].values

    step_array=np.diff(f_I_table.loc[:,"Frequency_Hz"])
    
    f_I_table = f_I_table.sort_values(by=["Cell_id", 'Stim_amp_pA'])
    f_I_table['Cell_id'] = pd.Categorical(f_I_table['Cell_id'])
    f_I_table['Sweep_number']=np.int64(f_I_table['Sweep_number'])

    f_I_table=f_I_table.sort_values(by=["Cell_id", 'Stim_amp_pA'])
    if np.count_nonzero(freq_array)==0 or np.count_nonzero(freq_array)<4 or np.count_nonzero(step_array)<3 :
        do_fit=False
        return f_I_table,do_fit
    
    else:
        do_fit=True
    return f_I_table,do_fit
    
    
    
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
        
def coarse_or_fine_sweep(cell_id):
    all_sweeps_table=pd.DataFrame(ctc.get_ephys_sweeps(cell_id))
    all_sweeps_table=all_sweeps_table[all_sweeps_table['stimulus_name']=="Long Square"]
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


def create_full_TPC_SF_table(cell_id,database):
    full_TPC_table=pd.DataFrame(columns=['Sweep','TPC'])
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
            
            
            
            stim_start_time=current_time_array[stim_start_index]
            stim_end_time=stim_start_time+1.
            TPC=ephys_treat.create_TPC(time_trace=current_time_array,
                                 potential_trace=current_membrane_trace,
                                 current_trace=current_stim_array,
                                 filter=2.)
            
            TPC=TPC[TPC['Time_s']<(stim_end_time+.05)]
            TPC=TPC[TPC['Time_s']>(stim_start_time-.05)]

            new_line=pd.Series([current_sweep,TPC],index=['Sweep','TPC'])
            full_TPC_table=full_TPC_table.append(new_line,ignore_index=True)
            
            stim_amp=my_Cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
            current_SF_table=ephys_treat.identify_spike(current_membrane_trace, current_time_array, current_stim_array, stim_start_time, stim_end_time)
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
            
            TPC=TPC[TPC['Time_s']<(stim_end_time+.05)]
            TPC=TPC[TPC['Time_s']>(stim_start_time-.05)]
            
            new_line=pd.Series([current_sweep,TPC],index=['Sweep','TPC'])
            full_TPC_table=full_TPC_table.append(new_line,ignore_index=True)
            current_SF_table=ephys_treat.identify_spike(current_membrane_trace, current_time_array, current_stim_array, stim_start_time, stim_end_time)
            new_SF_line=pd.Series([current_sweep,stim_start_time,stim_end_time,stim_amp,current_SF_table],index=['Sweep',"Stim_start_s","Stim_end_s",'Stim_amp_pA','SF'])
            full_SF_table=full_SF_table.append(new_SF_line,ignore_index=True)
    full_TPC_table.index=full_TPC_table.loc[:,'Sweep']

    full_SF_table.index=full_SF_table.loc[:,'Sweep']
    return full_TPC_table,full_SF_table
        

    
def create_metadata_table(cell_id,database,population_class,species_sweep_stim_table):
    metadata_table=pd.DataFrame(columns=["Database","Area","Layer","Input_Resistance_MOhm","Dendrite_type",'Min_stimulus',"Max_stim"])
    population_class.index=population_class.loc[:,"Cell_id"]
    if database=="Allen":
        cell_data=ctc.get_ephys_features(cell_id)
        cell_data=cell_data[cell_data["specimen_id"]==cell_id]
        cell_data.index=cell_data.loc[:,'specimen_id']
        IR=cell_data.loc[cell_id,"ri"]
        stim_freq_table,do_fit=extract_stim_freq(cell_id,species_sweep_stim_table,per_time=True,first_x_ms=5)
        min_stim=min(stim_freq_table.loc[:,'Stim_amp_pA'])
        max_stim=max(stim_freq_table.loc[:,'Stim_amp_pA'])
        database='Allen'
        population_class_line=population_class[population_class["Cell_id"]==str(cell_id)]


        Area=population_class.loc[str(cell_id),"General_area"]
        Layer=population_class_line.loc[str(cell_id),"Layer"]
        dendrite_type=population_class_line.loc[str(cell_id),"Dendrite_type"]
        
    new_line=pd.Series([database,
                        Area,
                        Layer,
                        IR,
                        dendrite_type,
                        min_stim,
                        max_stim],index=["Database","Area","Layer","Input_Resistance_MOhm","Dendrite_type",'Min_stimulus',"Max_stim"])
    metadata_table=metadata_table.append(new_line,ignore_index=True)
    return metadata_table
        


def create_cell_json(cell_id,database,population_class_file,species_sweep_stim_table):
    if database=='Allen':
        TPC,SF_table=create_full_TPC_SF_table(cell_id,database)
        
        cell_fit_table=pd.read_csv(str('/Users/julienballbe/My_Work/Allen_Data/Cell_Fit_table/'+str(cell_id)+'_Fit_table.csv'))
        cell_fit_table.index=cell_fit_table.loc[:,"Response_time_ms"]
        
        cell_feature_table=pd.read_csv(str('/Users/julienballbe/My_Work/Allen_Data/Cell_Features_table/'+str(cell_id)+'_Features_table.csv'))
        cell_feature_table.index=cell_feature_table.loc[:,'Response_time_ms']
        
        metadata_table=create_metadata_table(cell_id,database,population_class_file,species_sweep_stim_table)
        my_dict={'Metadata':[metadata_table],
                 'TPC':[TPC],
                 'Spike_feature_table':[SF_table],
                 'Fit_table':[cell_fit_table],
                 'IO_table':[cell_feature_table]}
        
    my_dict=pd.DataFrame(my_dict)
    my_dict.to_json(str('/Users/julienballbe/Downloads/Data_test_'+str(cell_id)+'.json'))
        
    
    
#%%
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
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    