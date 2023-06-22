#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:42:03 2023

@author: julienballbe
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from allensdk.core.cell_types_cache import CellTypesCache
import time
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle,geom_vline
import scipy
from scipy.stats import linregress
from scipy import optimize
from scipy.optimize import curve_fit
import random
from tqdm import tqdm

import logging
import warnings

import pandas

import os
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
from plotnine.labels import xlab
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, '/Users/julienballbe/My_Work/My_Librairies/Electrophy_treatment.py')
import Electrophy_treatment as ephys_treat
import Fit_library as fitlib
import Data_treatment as data_treat
import functools
import h5py
import glob
from datetime import date
import concurrent.futures
from multiprocessing import Pool
import time

#%%





#%%
def create_feature_vector_h5(cell_id_list, create_file=True,file_name='--',do_plot=False):
    
    
    first_ap_dv = pd.DataFrame([])
    first_ap_v = pd.DataFrame([])
    inst_freq = pd.DataFrame([])
    isi_shape_list = pd.DataFrame([])
    psth = pd.DataFrame([])
    spiking_fast_trough_v = pd.DataFrame([])
    spiking_peak_v = pd.DataFrame([])
    spiking_threshold_v = pd.DataFrame([])
    spiking_upstroke_downstroke_ratio = pd.DataFrame([])
    spiking_width = pd.DataFrame([])
    step_subthresh = pd.DataFrame([])
    subthresh_depol_norm = pd.DataFrame([])
    subthresh_norm_list = pd.DataFrame([])
    new_cell_id_list=pd.DataFrame([])
    problem_cell=[]
    
    
    for cell_id in tqdm(cell_id_list):
        
        
        try:
            Full_TPC_table,full_SF_dict_table,full_SF_table,Metadata_table,sweep_info_table,cell_fit_table,cell_feature_table,Adaptation_fit_table=data_treat.import_h5_file('/Volumes/Work_Julien/Cell_Data_File/Cell_'+str(cell_id)+'_data_file.h5',['All'])
            
    
            
            first_ap,first_ap_diff=create_first_AP_ss_ls_ramp_vector(cell_id,full_SF_table,Full_TPC_table,Metadata_table,sweep_info_table,do_plot=do_plot)
            isi_vector=create_isi_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,do_plot=do_plot)
        
            step_subthreshold_vector=create_step_subthreshold_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,target_amps=[-90,-70,-50,-30,-10],do_plot=do_plot)
            normalized_subthreshold_vector=create_normalized_subthreshold_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,do_plot=do_plot)
        
            psth_vector=create_psth_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,width=50,response_duration=.500,do_plot=do_plot)
        
            inst_freq_vector=create_inst_freq_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,width=20,response_duration=.500,normalized=False,do_plot=do_plot)
            norm_inst_freq_vector=create_inst_freq_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,width=20,response_duration=.500,normalized=True,do_plot=do_plot)
        
        
            up_down_binned_vector=create_binned_feature_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,feature='Up/Down',do_plot=do_plot)
            peak_binned_vector=create_binned_feature_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,feature='Peak',do_plot=do_plot)
            Fast_trough_binned_vector=create_binned_feature_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,feature='Fast_Trough',do_plot=do_plot)
            threshold_binned_vector=create_binned_feature_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,feature='Threshold',do_plot=do_plot)
            Width_binned_vector=create_binned_feature_vector(Full_TPC_table,full_SF_table,Metadata_table,sweep_info_table,feature='Width',do_plot=do_plot)    
            
            
            #first_ap_v=first_ap_v.append([first_ap],ignore_index=True)
            first_ap_v=pd.concat([first_ap_v,pd.DataFrame([first_ap])],ignore_index=True)
    
            #first_ap_dv=first_ap_dv.append([first_ap_diff],ignore_index=True)
            first_ap_dv=pd.concat([first_ap_dv,pd.DataFrame([first_ap_diff])],ignore_index=True)
            
            #isi_shape_list=isi_shape_list.append([isi_vector],ignore_index=True)
            isi_shape_list=pd.concat([isi_shape_list,pd.DataFrame([isi_vector])],ignore_index=True)
            
            #step_subthresh=step_subthresh.append([step_subthreshold_vector],ignore_index=True)
            step_subthresh = pd.concat([step_subthresh,pd.DataFrame([step_subthreshold_vector])],ignore_index=True)
            
            #subthresh_norm_list=subthresh_norm_list.append([normalized_subthreshold_vector],ignore_index=True)
            subthresh_norm_list=pd.concat([subthresh_norm_list,pd.DataFrame([normalized_subthreshold_vector])],ignore_index=True)
            
            #psth=psth.append([psth_vector],ignore_index=True)
            psth=pd.concat([psth,pd.DataFrame([psth_vector])],ignore_index=True)
        
            #inst_freq=inst_freq.append([inst_freq_vector],ignore_index=True)
            inst_freq=pd.concat([inst_freq,pd.DataFrame([inst_freq_vector])],ignore_index=True)
            
            #subthresh_depol_norm=subthresh_depol_norm.append([norm_inst_freq_vector],ignore_index=True)
            subthresh_depol_norm=pd.concat([subthresh_depol_norm,pd.DataFrame([norm_inst_freq_vector])],ignore_index=True)
            
            #spiking_upstroke_downstroke_ratio=spiking_upstroke_downstroke_ratio.append([up_down_binned_vector],ignore_index=True)
            spiking_upstroke_downstroke_ratio=pd.concat([spiking_upstroke_downstroke_ratio,pd.DataFrame([up_down_binned_vector])],ignore_index=True)
            
            #spiking_peak_v=spiking_peak_v.append([peak_binned_vector],ignore_index=True)
            spiking_peak_v=pd.concat([spiking_peak_v,pd.DataFrame([peak_binned_vector])],ignore_index=True)
            
            #spiking_fast_trough_v=spiking_fast_trough_v.append([Fast_trough_binned_vector],ignore_index=True)
            spiking_fast_trough_v=pd.concat([spiking_fast_trough_v,pd.DataFrame([Fast_trough_binned_vector])],ignore_index=True)
            
            #spiking_threshold_v=spiking_threshold_v.append([threshold_binned_vector],ignore_index=True)
            spiking_threshold_v=pd.concat([spiking_threshold_v,pd.DataFrame([threshold_binned_vector])],ignore_index=True)
            
            #spiking_width=spiking_width.append([Width_binned_vector],ignore_index=True)
            spiking_width=pd.concat([spiking_width,pd.DataFrame([Width_binned_vector])],ignore_index=True)
            
            #new_cell_id_list=new_cell_id_list.append([cell_id],ignore_index=True)
            new_cell_id_list=pd.concat([new_cell_id_list,pd.DataFrame([cell_id])],ignore_index=True)
        
        except Exception as e:
            problem_cell.append(str(cell_id))
            print(str('Problem_with_cell:'+str(cell_id)+str(e)))
            
        
    
    if create_file==True:
        
        f = h5py.File(str("/Users/julienballbe/My_Work/Clustering/Feature_files/feature_vector"+str(file_name)+".h5", "a"))
        f["first_ap_v"] = first_ap_v

        f["first_ap_dv"] = first_ap_dv
        f["ids"] = np.array(cell_id_list)

        f["inst_freq"] = inst_freq
        f["isi_shape"] = isi_shape_list
        f["psth"] = psth

        f["spiking_fast_trough_v"] = spiking_fast_trough_v
        f["spiking_peak_v"] = spiking_peak_v
        f["spiking_threshold_v"] = spiking_threshold_v
        f["spiking_upstroke_downstroke_ratio"] = spiking_upstroke_downstroke_ratio
        f["spiking_width"] = spiking_width
        f["step_subthresh"] = step_subthresh
        f["subthresh_depol_norm"] = subthresh_depol_norm
        f["subthresh_norm"] = subthresh_norm_list
        f.close()
        
        return problem_cell
    else:
        return first_ap_dv,first_ap_v,new_cell_id_list,inst_freq,isi_shape_list,psth,spiking_fast_trough_v,spiking_peak_v,spiking_threshold_v,spiking_upstroke_downstroke_ratio,spiking_width,step_subthresh,subthresh_depol_norm,subthresh_norm_list,problem_cell
    
    

#%%

def create_first_AP_ss_ls_ramp_vector(cell_id , original_full_SF_table,original_full_TPC_table,
                                      original_metadata_table,original_sweep_info_table,target_sampling_rate=10000, window_length=0.003,do_plot=False):
    
    
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table = original_full_TPC_table.copy()
    metadata_table = original_metadata_table.copy()
    sweep_info_table = original_sweep_info_table.copy()
    
    if metadata_table.loc[0,'Database']=='Allen_Cell_Type_Database':
        cell_id=int(cell_id)
    
    
    ss_sweep,ls_sweep,ramp_sweep=identify_sweep_for_first_ap_waveform(cell_id,full_SF_table, metadata_table,sweep_info_table)
    
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
        x1=time_array[len(ss_ap_wf)]
        x2=time_array[len(ss_ap_wf)+len(ls_ap_wf)]
        
        
        time_array_diff=np.arange(0, len(first_ap_diff)) * (1.0 / sampling_rate)
        x1_diff=time_array_diff[len(ss_ap_wf_diff)]
        x2_diff=time_array_diff[len(ss_ap_wf_diff)+len(ls_ap_wf_diff)]
        
        ap_wf_table=pd.concat([pd.Series(first_ap),pd.Series(time_array)],axis=1)
        ap_wf_table.columns=["Membrane_potential_mV","Time_s"]
        
        ap_wf_diff_table=pd.concat([pd.Series(first_ap_diff),pd.Series(time_array_diff)],axis=1)
        ap_wf_diff_table.columns=["Membrane_potential_derivative_mV/s","Time_s"]
        
        
        first_ap_plot=ggplot(ap_wf_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        first_ap_plot+=ggtitle('First_AP_waveform')
        first_ap_plot+=geom_vline(xintercept=[x1,x2],colour=['red','red'])
        print(first_ap_plot)
        
        first_ap_diff_plot=ggplot(ap_wf_diff_table,aes(x='Time_s',y="Membrane_potential_derivative_mV/s"))+geom_line()
        first_ap_diff_plot+=ggtitle('First_AP_derivative_waveform')
        first_ap_diff_plot+=geom_vline(xintercept=[x1_diff,x2_diff],colour=['red','red'])
        print(first_ap_diff_plot)
        
    return first_ap, first_ap_diff


def identify_sweep_for_first_ap_waveform(cell_id,original_full_SF_table, original_metadata_table,original_sweep_info_table):
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
    
    sweep_info_table=original_sweep_info_table.copy()
    
    if metadata_table.loc[0,'Database']=='Allen_Cell_Type_Database':
        cell_id=int(cell_id)
        ctc=CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        allen_sweep_stim_table=pd.read_pickle('/Users/julienballbe/My_Work/Allen_Data/Allen_ID_Sweep_stim_table.pkl')
        cell_data=ctc.get_ephys_data(cell_id)

        ### For allen Data, deal with short square stimulus
        ss_coarse_sweep_table=data_treat.coarse_or_fine_sweep(cell_id,stimulus_type='Short Square')
        ss_coarse_sweep_table=ss_coarse_sweep_table[ss_coarse_sweep_table['stimulus_description']=='COARSE']
        
        ss_coarse_sweep_table.index=ss_coarse_sweep_table['sweep_number']
        ss_coarse_sweep_table['Stim_amp_pA']=0.
        ss_coarse_sweep_table['Nb_of_Spike']=0
        for current_sweep in np.array(ss_coarse_sweep_table['sweep_number']):
            stim_amp=cell_data.get_sweep_metadata(current_sweep)['aibs_stimulus_amplitude_pa']
            
            ss_coarse_sweep_table.loc[current_sweep,'Stim_amp_pA']=stim_amp
            
            nb_of_spike=len(cell_data.get_spike_times(current_sweep))
           
            ss_coarse_sweep_table.loc[current_sweep,'Nb_of_Spike']=nb_of_spike
        ss_coarse_sweep_table=ss_coarse_sweep_table[ss_coarse_sweep_table['Nb_of_Spike']!=0]
        
        if ss_coarse_sweep_table.shape[0]==0:
            ss_coarse_sweep_table=data_treat.coarse_or_fine_sweep(cell_id,stimulus_type='Short Square')
            ss_coarse_sweep_table=ss_coarse_sweep_table[ss_coarse_sweep_table['stimulus_description']=='FINEST']
            ss_coarse_sweep_table.index=ss_coarse_sweep_table['sweep_number']
            ss_coarse_sweep_table['Stim_amp_pA']=0.
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
            

    sweep_list=np.array(sweep_info_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocol'].values[0])

    Nb_of_Traces=full_SF_table.shape[0]
    
    
    first_ap_ls_sweep=np.array([])
    
    for current_sweep in sweep_list:
        current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF']).copy()
        current_SF=current_SF[current_SF['Feature']=='Threshold']
        current_SF=current_SF.sort_values(by=['Time_s'])
        current_spike_time = np.array(current_SF['Time_s'])
        nb_of_spike=current_SF[current_SF['Feature']=='Threshold'].shape[0]
        
        current_end_stim_time=sweep_info_table.loc[current_sweep,'Stim_end_s']

        if current_spike_time.size!=0:
            if current_spike_time[0]+0.003 < current_end_stim_time:

                first_ap_ls_sweep=np.array([current_sweep])
                break
        
    return first_ap_ss_sweep,first_ap_ls_sweep,first_ap_ramp_sweep
    
    # for elt in range(Nb_of_Traces//Nb_of_Protocols):
    #     current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
    #     sweep_zero_spike_check=1
    #     #current_max_nb_of_spike=0
    #     for current_idx in current_sweeps_idx:
    #         current_sweep=sweep_list[current_idx]
    #         current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF']).copy()
    #         current_SF=current_SF[current_SF['Feature']=='Threshold']
    #         current_SF=current_SF.sort_values(by=['Time_s'])
    #         current_spike_time = np.array(current_SF['Time_s'])
    #         nb_of_spike=current_SF[current_SF['Feature']=='Threshold'].shape[0]
            
    #         current_end_stim_time=sweep_info_table.loc[current_sweep,'Stim_end_s']
            
    #         if current_spike_time.size==0:
    #             sweep_zero_spike_check*=current_spike_time.size
            
    #         elif current_spike_time[0]+0.003 < current_end_stim_time: #check if the 3ms window after threshold is not cut by stim end
    #             sweep_zero_spike_check*=current_spike_time.size
    #             #current_min_nb_of_spike=min(current_max_nb_of_spike,nb_of_spike)
                
    #         elif current_spike_time[0]+0.003 >= current_end_stim_time:
    #             sweep_zero_spike_check*=0 # if the 3ms window after threshold is cut by stim end, do not consider the spike
                
        
    #     if sweep_zero_spike_check>=1: #in case multiple sweeps have the same stim amplitude, check that each sweep has at least one spike
    #         first_ap_ls_sweep=sweep_list[current_sweeps_idx]
    #         break
   
    #return first_ap_ss_sweep,first_ap_ls_sweep,first_ap_ramp_sweep
    
    

def first_ap_vectors(cell_id ,sweeps_list, protocol='ss' ,original_Full_SF_table=None, original_Full_TPC_table=None,
        target_sampling_rate=20000, window_length=0.003,
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
            
        avg_ap = np.vstack(ap_list).mean(axis=0) #Do the average of first AP if there are multiple sweeps at the same stimulus amplitude   
        
        if sampling_rate > target_sampling_rate:
            #print('original_len',len(avg_ap))

            #print(target_sampling_rate,sampling_rate)
            sampling_factor = int(sampling_rate // target_sampling_rate)
            avg_ap = _subsample_average(avg_ap, sampling_factor)
            #print('subsampled_len',len(avg_ap))
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
    start_index=ephys_treat.find_time_index(time_trace, first_threshold_time)
    end_index = start_index + length_in_points
    return potential_trace[start_index:end_index]
        
def _subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg


def create_isi_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,duration=.500, min_spike=5,do_plot=False):
    Full_TPC_table=original_Full_TPC_table.copy()
    Full_SF_table=original_Full_SF_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    metadata_table=original_metadata_table.copy()
    
    selected_isi_sweep=identify_sweep_for_isi_shape(Full_SF_table,metadata_table,sweep_info_table,duration=duration,min_spike=min_spike)
    
    isi_norm=get_isi_norm(selected_isi_sweep,Full_SF_table,Full_TPC_table,sweep_info_table,do_plot=do_plot)
    
    return isi_norm
    
    
def identify_sweep_for_isi_shape(original_full_SF_table, original_metadata_table,original_sweep_info_table,duration=.500, min_spike=5):
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
    
    sweep_info_table=original_sweep_info_table.copy()
    
    sweep_list=np.array(sweep_info_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocol'].values[0])
    Nb_of_Traces=full_SF_table.shape[0]
    max_nb_spike=0
    for sweep in sweep_list:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[sweep,'SF'])

        max_nb_spike=max(max_nb_spike,sub_SF_table[sub_SF_table['Feature']=='Peak'].shape[0])
    
    selected_sweep=[]

    for current_sweep in sweep_list:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[current_sweep,'SF']).copy()
        nb_spike=sub_SF_table[sub_SF_table['Feature']=='Peak'].shape[0]
        
        if max_nb_spike>=min_spike:
            if nb_spike>=5:
                selected_sweep=np.array([current_sweep])
                break
        elif max_nb_spike>1:
            if nb_spike==max_nb_spike:
                nb_spike=np.array([current_sweep])
                break
        else:
            if max_nb_spike==1:
                nb_spike=np.array([current_sweep])
                break
        

    # for elt in range(Nb_of_Traces//Nb_of_Protocols):
    #     current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
    #     current_max_nb_of_spike=0
    #     for current_idx in current_sweeps_idx:
    #         current_sweep=sweep_list[current_idx]
    #         current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
    #         nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
    #         current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
        
    #     if max_nb_spike>=min_spike:
    #         if current_max_nb_of_spike>=5:
    #             selected_sweep=np.array(sweep_list[current_sweeps_idx])
    #             break
    #     elif max_nb_spike>1:
    #         if current_max_nb_of_spike==max_nb_spike:
    #             selected_sweep=np.array(sweep_list[current_sweeps_idx])
    #             break
    #     else:
    #         if current_max_nb_of_spike==1:
    #             selected_sweep=np.array(sweep_list[current_sweeps_idx])
    #             break
           
    return selected_sweep


def get_isi_norm(sweep_list,original_full_SF_table, original_full_TPC_table,original_sweep_info_table,response_duration=.500,do_plot=False):
    full_SF_table=original_full_SF_table.copy()
    full_TPC_table=original_full_TPC_table.copy()
    sweep_info_table=original_sweep_info_table.copy()

    
    if len(sweep_list)==1:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[sweep_list[0],'SF'])
        sub_TPC_table=pd.DataFrame(full_TPC_table.loc[sweep_list[0],'TPC'])
        stim_start_time=sweep_info_table.loc[sweep_list[0],'Stim_start_s']
        stim_end_time=sweep_info_table.loc[sweep_list[0],'Stim_end_s']
        #stim_end_time=stim_start_time+response_duration
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
            stim_start_time=sweep_info_table.loc[sweep_list[0],'Stim_start_s']
            stim_end_time=sweep_info_table.loc[sweep_list[0],'Stim_end_s']
            #stim_end_time=stim_start_time+response_duration
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
        stim_end_index = ephys_treat.find_time_index(np.array(TPC_table.loc[:,'Time_s']), end)+TPC_table.index[0]

        if fast_trough_t < end - steady_state_interval:
            
            max_end_index = ephys_treat.find_time_index(np.array(TPC_table.loc[:,'Time_s']), fast_trough_time + single_max_duration)+TPC_table.index[0]
            
            std_start_index = ephys_treat.find_time_index(np.array(TPC_table.loc[:,'Time_s']), end - steady_state_interval)+TPC_table.index[0]
            
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

            
def create_step_subthreshold_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,target_amps=[-90,-70,-50,-30,-10],do_plot=False):
    
    full_SF_table=original_Full_SF_table.copy()
    full_TPC_table=original_Full_TPC_table.copy()
    Metadata_table=original_metadata_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    selected_sub_threshold_sweep=identify_subthreshold_hyperpol_with_amplitudes(full_TPC_table,
                                                                                full_SF_table,
                                                                                Metadata_table,
                                                                                sweep_info_table)
    
    subthreshold_array=step_subthreshold(selected_sub_threshold_sweep,
                                         full_TPC_table,
                                         original_Full_SF_table,
                                         sweep_info_table,
                                         target_amps,do_plot=do_plot)
    
    return subthreshold_array
    
    
def identify_subthreshold_hyperpol_with_amplitudes(original_full_TPC_table,original_full_SF_table, original_Metadata_table,original_sweep_info_table):
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
    sweep_info_table=original_sweep_info_table.copy()
    
    sweep_list=np.array(sweep_info_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    Nb_of_Protocols=int(Metadata_table.loc[:,'Nb_of_Protocol'].values[0])
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
            current_stim_start=sweep_info_table.loc[current_subthresh_sweeps,'Stim_start_s']
            
            potential_baseline=np.mean(np.array(current_TPC[current_TPC['Time_s']<(current_stim_start-.005)]['Membrane_potential_mV']))
            after_stim_potential_table=current_TPC[current_TPC['Time_s']>(current_stim_start+.005)]
            after_stim_potential_table=after_stim_potential_table[after_stim_potential_table['Time_s']<current_stim_start+.5]
            potential_after_stim_start=np.mean(np.array(after_stim_potential_table['Membrane_potential_mV']))
           
            if potential_baseline < potential_after_stim_start: #if current_sweep is depolarizing
                depolarizing_sweep+=1
                
        if depolarizing_sweep == 0: # if no sweep in the sublist (in case of duplicate/triplicate) is depolarizing --> hyperpolarizing

            sub_threshold_hyperpol_sweeps.append(current_subthresh_sweeps_list)
            
    
            
     
    return np.array(sub_threshold_hyperpol_sweeps)



def step_subthreshold(selected_sweep_list,original_full_TPC_table,original_full_SF_table,original_sweep_info_table, target_amps,
                      extend_duration=0.2, subsample_interval=0.01,
                      amp_tolerance=0.,
                      do_plot=False):
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

   
    
    if len(selected_sweep_list)==0:
        if do_plot:
            step_subthreshold_plot=ggplot()
            step_subthreshold_plot+=ggtitle('No_sweeps_for_target_amplitude')
            print(step_subthreshold_plot)
        
        return np.nan
            
    

    full_TPC_table=original_full_TPC_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    extend_length = int(np.round(extend_duration / subsample_interval))
    subsampled_dict = {}
    
    for current_sweep_sub_list in selected_sweep_list:
        mean_amp=np.mean(sweep_info_table.loc[current_sweep_sub_list,'Stim_amp_pA'])
        potential_list=[]
        for current_sweep in current_sweep_sub_list:
            
            current_TPC=pd.DataFrame(full_TPC_table.loc[current_sweep,'TPC'])
            current_TPC=current_TPC.reset_index(drop=True)
            

            current_time_array=np.array(current_TPC['Time_s'])
            current_potential_array = np.array(current_TPC['Membrane_potential_mV'])
            
            current_stim_start=sweep_info_table.loc[current_sweep,'Stim_start_s']
            current_start_index=ephys_treat.find_time_index(current_time_array, current_stim_start - extend_duration)
            
            current_stim_end=sweep_info_table.loc[current_sweep,'Stim_end_s']
            current_end_index=ephys_treat.find_time_index(current_time_array, current_stim_end + extend_duration)

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
            
    output_vector=np.hstack(output_list) 
    if do_plot:
        step_subthreshold_table=pd.DataFrame({'Length_in_points':np.arange(len(output_vector)),
                                              'Membrane_potential_mV':output_vector})
        step_subthreshold_plot=ggplot(step_subthreshold_table,aes(x='Length_in_points',y='Membrane_potential_mV'))+geom_line()
        step_subthreshold_plot+=geom_vline(xintercept=list(np.arange(0,len(output_vector),step=len(output_vector)//len(target_amps))),color='red')
        step_subthreshold_plot+=ggtitle('Concatenated_subthreshold_responses')
        print(step_subthreshold_plot)
    return output_vector

def create_normalized_subthreshold_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,extend_duration=0.2, subsample_interval=0.01,do_plot=False):
    
    Full_TPC_table=original_Full_TPC_table.copy()
    Full_SF_table=original_Full_SF_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    metadata_table=original_metadata_table.copy()
    
    norm_subthreth_sweep=identify_sweep_for_norm_subthreshold(Full_SF_table,metadata_table,sweep_info_table)
    
    subthreshold_vector=subthresh_norm(norm_subthreth_sweep,Full_TPC_table,sweep_info_table,extend_duration=extend_duration,subsample_interval=subsample_interval,do_plot=do_plot)
    
    return subthreshold_vector


def subthresh_norm(selected_sweep_list,original_full_TPC_table ,original_sweep_info_table,#Need to adapt to get minimum current amplitude
                   extend_duration=0.2, subsample_interval=0.01,do_plot=False):
    """ Response to largest amplitude hyperpolarizing current step amplitude aligned to baseline potential and normalized by peak deflection

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
    

    full_TPC_table=original_full_TPC_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    extend_length = int(np.round(extend_duration / subsample_interval)) # represent extended duration in subsampled scale

    
    
    minimum_amplitude=np.inf
    for current_sweep_sub_list in selected_sweep_list:
        current_mean_amp=np.mean(sweep_info_table.loc[current_sweep_sub_list,'Stim_amp_pA'])

        if current_mean_amp<minimum_amplitude:
            minimum_amplitude=current_mean_amp

            potential_list=[]
            for current_sweep in current_sweep_sub_list:
                
                current_TPC=pd.DataFrame(full_TPC_table.loc[current_sweep,'TPC'])
                current_TPC=current_TPC.reset_index(drop=True)
                
    
                current_time_array=np.array(current_TPC['Time_s'])
                current_potential_array = np.array(current_TPC['Membrane_potential_mV'])
                
                current_stim_start=sweep_info_table.loc[current_sweep,'Stim_start_s']
                current_start_index=ephys_treat.find_time_index(current_time_array, current_stim_start - extend_duration) # Strat time correspind to stim start - extended duration
                
                current_stim_end=sweep_info_table.loc[current_sweep,'Stim_end_s']
                current_end_index=ephys_treat.find_time_index(current_time_array, current_stim_end + extend_duration)
    
                delta_t = current_time_array[1] - current_time_array[0]
                subsample_width = int(np.round(subsample_interval / delta_t))
                current_subsampled_potential = _subsample_average(current_potential_array[current_start_index:current_end_index], subsample_width)
                
                
                potential_list.append(current_subsampled_potential)
                
                
            minimum_amp_potential = np.vstack(potential_list).mean(axis=0)
            base_v = minimum_amp_potential[:extend_length].mean() #base = baseline potential before stim start
            peak_deflect=minimum_amp_potential[extend_length:-extend_length].min() # take minimum value between actual stimulus start and stimulus end
            delta=base_v-peak_deflect
            
            normalized_minimum_amp_potential = minimum_amp_potential-base_v
            normalized_minimum_amp_potential /= delta
    if do_plot:
        x_data=pd.Series(np.arange(0,len(normalized_minimum_amp_potential)))
        y_data=pd.Series(normalized_minimum_amp_potential)
        full_table=pd.DataFrame({'Length_in_point':x_data,
                                 'Normalized_potential':y_data})
        norm_subthreshold_plot=ggplot(full_table,aes(x='Length_in_point',y="Normalized_potential"))+geom_line()
        norm_subthreshold_plot+=ggtitle('Normalized_largest_amplitude_subthreshold')
        print(norm_subthreshold_plot)
        #plt.plot(np.array(normalized_minimum_amp_potential))
    return normalized_minimum_amp_potential






def identify_sweep_for_norm_subthreshold(original_full_SF_table, original_metadata_table,original_sweep_info_table):
    """ Find lowest-amplitude non-spiking sweep 

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
    
    full_SF_table=original_full_SF_table.copy()

    metadata_table=original_metadata_table.copy()
    
    sweep_info_table=original_sweep_info_table.copy()
    
    sweep_list=np.array(sweep_info_table.sort_values(by=['Stim_amp_pA'])['Sweep']) # sort sweeps per increasing stimulus amplitude

    


    selected_sweep=[]
    for current_sweep in sweep_list:
        sub_SF_table=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
        nb_spike=sub_SF_table[sub_SF_table['Feature']=='Peak'].shape[0]
        if nb_spike==0:

            selected_sweep=[current_sweep]
            break
            
       
    # Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocol'].values[0])
    # Nb_of_Traces=full_SF_table.shape[0]
    # for elt in range(Nb_of_Traces//Nb_of_Protocols):
    #     current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
    #     current_max_nb_of_spike=0
    #     for current_idx in current_sweeps_idx:
    #         current_sweep=sweep_list[current_idx]
    #         current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
    #         nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
            
    #         current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
        
    #     if current_max_nb_of_spike==0:
    #         selected_sweep=np.array(sweep_list[current_sweeps_idx])
                
           
    return np.array([selected_sweep])


def identify_sweep_for_psth_inst_freq_binned_features(original_full_SF_table, original_full_TPC_table, original_metadata_table,original_sweep_info_table,stim_amp_step=20, stim_amp_step_margin=5):
    '''
    Identify sweeps from rheobase (lowest stimulus amplitude with at least 1 AP) to rheobase+100pA with a step of stim_amp_step (default=20pA)

    Parameters
    ----------
    original_full_SF_table : DataFrame
        DESCRIPTION.
    original_full_TPC_table : DataFrame
        DESCRIPTION.
    original_metadata_table : DataFrame
        DESCRIPTION.
    original_sweep_info_table : DataFrame
        DESCRIPTION.
    stim_amp_step : Float, optional
        Current amplitude step between the sweeps. The default is 20.
    stim_amp_step_margin : float, optional
        Stimulus amplitude margin to search correspondign sweep. search sweep whic curretn amplitude = target amplitude ± stim_amp_step_margin . The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    full_SF_table=original_full_SF_table.copy()
    
    metadata_table=original_metadata_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    sweep_list=np.array(sweep_info_table.sort_values(by=['Stim_amp_pA'])['Sweep'])

    

    
    selected_sweeps_psth=[]
    identified_rheobase=False
    
    
    for current_sweep in sweep_list: #Loop to identify first sweep with at least one spike= rheobase
        current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
        nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]

        if nb_of_spike!=0 and identified_rheobase==False:
            identified_rheobase=True
            rheobase=sweep_info_table.loc[current_sweep,:]['Stim_amp_pA']
            theorical_target_amps=list(np.arange(rheobase,rheobase+101,stim_amp_step))
            selected_sweeps_psth.append(np.array([current_sweep]))
            break
    
    for current_target_amp in theorical_target_amps[1:]:
        lowest_amp=current_target_amp-stim_amp_step_margin
        highest_amp=current_target_amp+stim_amp_step_margin
        identified_sweep_for_amp=False

        for current_sweep in sweep_list:
            if identified_sweep_for_amp==True:
                break
            current_stim_amp=sweep_info_table.loc[current_sweep,:]['Stim_amp_pA']
            current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
            nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
            if current_stim_amp >= lowest_amp and current_stim_amp <= highest_amp and nb_of_spike != 0 and identified_sweep_for_amp==False:
                selected_sweeps_psth.append(np.array([current_sweep]))
                identified_sweep_for_amp=True
                
        if identified_sweep_for_amp==False:
            selected_sweeps_psth.append(np.empty(1,dtype=object))
            
        
    return np.array(selected_sweeps_psth)
            
    # Nb_of_Protocols=int(metadata_table.loc[:,'Nb_of_Protocol'].values[0])
    # Nb_of_Traces=full_SF_table.shape[0]
    # end_rheobase=0
    # previous_stim_amp_avg=0         
    # for elt in range(Nb_of_Traces//Nb_of_Protocols):
    #     current_sweeps_idx=np.arange(elt*Nb_of_Protocols,elt*Nb_of_Protocols+Nb_of_Protocols)
    #     current_max_nb_of_spike=0
    #     for current_idx in current_sweeps_idx:
    #         current_sweep=sweep_list[current_idx]
    #         current_SF=pd.DataFrame(full_SF_table.loc[current_sweep,'SF'])
    #         nb_of_spike=current_SF[current_SF['Feature']=='Peak'].shape[0]
    #         current_max_nb_of_spike=max(current_max_nb_of_spike,nb_of_spike)
    #     current_stim_amp_avg=np.mean(np.array(sweep_info_table.loc[sweep_list[current_sweeps_idx],:]['Stim_amp_pA']))
        
    #     if current_max_nb_of_spike>=1 and identified_rheobase == False:
    #         identified_rheobase=True
    #         rheobase=current_stim_amp_avg
    #         selected_sweeps_psth.append(np.array(sweep_list[current_sweeps_idx]))
    #         end_rheobase=rheobase+100
    #         previous_stim_amp_avg=rheobase
        
    #     elif current_stim_amp_avg <= end_rheobase and identified_rheobase==True:
    #         #if in the protocol, the stimulus step is higher than 20pA(stim_amp_step), 
    #         #add false sweep id, to interpolate in the next function
    #         #add a 5pA error margin to avoid false 20pA-steps i.e.: (-24.344--64.643)//20
    #         step_amp=int((current_stim_amp_avg-previous_stim_amp_avg)//(stim_amp_step+stim_amp_step_margin)) 
            
    #         for current_step in range(step_amp): # if step_amp==0 (so no 20pA step skipped), the loop is null, so nothing is added
    #             selected_sweeps_psth.append(np.zeros_like(current_sweeps_idx))
    #         selected_sweeps_psth.append(np.array(sweep_list[current_sweeps_idx]))
    #         previous_stim_amp_avg=current_stim_amp_avg
            
    #     elif current_stim_amp_avg > end_rheobase and identified_rheobase==True:
    #         break
           
    #return np.array(selected_sweeps_psth)


def get_psth_vector(selected_sweeps_psth,original_full_SF_table,original_sweep_info_table, width=50,response_duration=.500,do_plot=False):
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
    sweep_info_table=original_sweep_info_table.copy()
    vector_list = []
    nb_of_sweeps=0
    for current_sweep_list in selected_sweeps_psth:
        output=[]
        if current_sweep_list[0] == None:
            vector_list.append(None)
            continue
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=SF_table.loc[current_sweep,'SF']
            current_stim_start=sweep_info_table.loc[current_sweep,'Stim_start_s']
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
        nb_of_sweeps+=1
        
    output_psth_vector = _combine_and_interpolate(vector_list)
    if do_plot:
        psth_df=pd.DataFrame({'Bin_index':np.arange(len(output_psth_vector)),
                              'Frequency_Hz':output_psth_vector})
        psth_df['Bin_index']+=.5
        psth_plot=ggplot()+geom_point(psth_df,aes(x="Bin_index",y="Frequency_Hz"),shape='_',size=3,stroke=.8)
        psth_plot+=geom_vline(xintercept=list(np.arange(0,len(output_psth_vector),step=len(output_psth_vector)//nb_of_sweeps)),color='red')
        psth_plot+=ggtitle('PSTH')
        print(psth_plot)
        
        
    return output_psth_vector

def create_psth_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,width=50,response_duration=.500,do_plot=False):
    full_SF_table=original_Full_SF_table.copy()
    full_TPC_table=original_Full_TPC_table.copy()
    metadata_table=original_metadata_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    
    selected_sweeps=identify_sweep_for_psth_inst_freq_binned_features(full_SF_table, full_TPC_table, metadata_table, original_sweep_info_table)

    psth=get_psth_vector(selected_sweeps,full_SF_table,sweep_info_table,width=width,response_duration=response_duration,do_plot=do_plot)
    return psth
    
def create_inst_freq_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,width=20,response_duration=.500,normalized=False,do_plot=False):
    full_SF_table=original_Full_SF_table.copy()
    full_TPC_table=original_Full_TPC_table.copy()
    metadata_table=original_metadata_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    
    selected_sweeps=identify_sweep_for_psth_inst_freq_binned_features(full_SF_table, full_TPC_table, metadata_table, original_sweep_info_table)
    
    inst_freq=get_inst_freq_vector(selected_sweeps, full_SF_table,sweep_info_table,width=width,response_duration=response_duration,normalized=normalized,do_plot=do_plot)
    return inst_freq
    
def get_inst_freq_vector(selected_sweeps_inst_freq,original_full_SF_table, original_sweep_info_table,width=20,response_duration=.500,normalized=False,do_plot=False):
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
    sweep_info_table=original_sweep_info_table.copy()
    vector_list = []

    for current_sweep_list in selected_sweeps_inst_freq:
        output=[]

        if current_sweep_list[0] == None:
            vector_list.append(None)
            continue
        
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=SF_table.loc[current_sweep,'SF']
            current_stim_start=sweep_info_table.loc[current_sweep,'Stim_start_s']
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

        
    output_inst_freq_vector = _combine_and_interpolate(vector_list)

    if do_plot:
        inst_freq_df=pd.DataFrame({"Bin_index":np.arange(len(output_inst_freq_vector)),
                                   "Inst_freq_Hz":output_inst_freq_vector})
        inst_freq_df['Bin_index']+=.5
        inst_freq_plot=ggplot()+geom_point(inst_freq_df,aes(x="Bin_index",y="Inst_freq_Hz"),shape='_',size=1,stroke=.7)
        inst_freq_plot+=geom_vline(xintercept=list(np.arange(0,len(output_inst_freq_vector),step=len(output_inst_freq_vector)//len(selected_sweeps_inst_freq))),color='red')
        inst_freq_plot+=ggtitle('Instantaneous_Frequency')
        print(inst_freq_plot)
        

    return output_inst_freq_vector

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




def create_binned_feature_vector(original_Full_TPC_table,original_Full_SF_table,original_metadata_table,original_sweep_info_table,feature='Peak',width=20,response_duration=.500,do_plot=False):

    full_SF_table=original_Full_SF_table.copy()
    full_TPC_table=original_Full_TPC_table.copy()
    metadata_table=original_metadata_table.copy()
    sweep_info_table=original_sweep_info_table.copy()
    
    
    selected_sweeps=identify_sweep_for_psth_inst_freq_binned_features(full_SF_table, full_TPC_table, metadata_table, original_sweep_info_table)
    
    binned_feature_vector=binned_AP_single_feature(selected_sweeps,full_SF_table,full_TPC_table,sweep_info_table,feature=feature,width=width,response_duration=response_duration,do_plot=do_plot)
    
    return binned_feature_vector

def binned_AP_single_feature(selected_sweeps_inst_freq,original_full_SF_table, original_full_TPC_table, original_sweep_info_table ,feature='Peak',width=20,response_duration=.500,do_plot=False):
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
    sweep_info_table=original_sweep_info_table.copy()
    vector_list = []
    Original_data_vector=[]
    n_bins=int(response_duration*1e3//width)
    for current_sweep_list in selected_sweeps_inst_freq:
        output=[]
        sweep_col=[]
        bin_columns=[]
        Stim_table=pd.DataFrame(columns=['Bin_index',str(feature),'Original_Data','Sweep'])
        if current_sweep_list[0] == None:
            vector_list.append(None)
            Original_data_vector.append(np.ones(n_bins,dtype=bool))
            continue
        
        for elt in range(len(current_sweep_list)):
            current_sweep=current_sweep_list[elt]
            current_SF_table=pd.DataFrame(SF_table.loc[current_sweep,'SF'].copy())
            current_stim_start=sweep_info_table.loc[current_sweep,'Stim_start_s']
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
            # includes right edge, so adding one to desired bin number
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
        
        
    output_vector = _combine_and_interpolate(vector_list)

    if do_plot:

        Original_data_vector=np.hstack(Original_data_vector)
        Full_table=pd.DataFrame()
        Full_table['Bin_index']=pd.Series(np.arange(len(output_vector)))
        Full_table[str(feature)]=pd.Series(output_vector)
        Full_table['Extrapolated_sweep']=pd.Series(Original_data_vector)
        my_plot=ggplot(Full_table,aes(x='Bin_index',y=Full_table.iloc[:,1],color='Extrapolated_sweep'))+geom_point()
        my_plot+=geom_vline(xintercept=(np.arange(len(selected_sweeps_inst_freq))*n_bins)-1)

        
        if feature=='Up/Down':

            my_plot+=ggtitle("Upstroke/Downstroke_ratio")
        elif feature=='Width':
            my_plot+=ggtitle('AP_width_s')
        else:
            my_plot+=ggtitle(str(feature+'_potential_mV'))
            
        print(my_plot)
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
  

    width_levels = np.zeros_like(trough_time_array) * np.nan
    width_levels[use_indexes] = heights[use_indexes] / 2. + trough_potential_array[use_indexes]
   


    thresh_to_peak_levels = np.zeros_like(trough_time_array) * np.nan
    thresh_to_peak_levels[use_indexes] = (peak_potential_array[use_indexes] - threshold_potential_array[use_indexes]) / 2. + threshold_potential_array[use_indexes]
   

    # Some spikes in burst may have deep trough but short height, so can't use same
    # definition for width

    width_levels[width_levels < threshold_potential_array] = thresh_to_peak_levels[width_levels < threshold_potential_array]

    width_starts = np.zeros_like(trough_time_array) * np.nan
    width_starts[use_indexes] = np.array([pk - np.flatnonzero(potential_trace[pk:spk:-1] <= wl)[0] if
                    np.flatnonzero(potential_trace[pk:spk:-1] <= wl).size > 0 else np.nan for pk, spk, wl
                    in zip(peak_index[use_indexes], threshold_index[use_indexes], width_levels[use_indexes])])
    
    width_ends = np.zeros_like(trough_time_array) * np.nan

    width_ends[use_indexes] = np.array([pk + np.flatnonzero(potential_trace[pk:tr] <= wl)[0] if
                    np.flatnonzero(potential_trace[pk:tr] <= wl).size > 0 else np.nan for pk, tr, wl
                    in zip(peak_index[use_indexes], trough_index[use_indexes].astype(int), width_levels[use_indexes])])
    
    missing_widths = np.isnan(width_starts) | np.isnan(width_ends)
    widths = np.zeros_like(width_starts, dtype=np.float64)
    widths[~missing_widths] = time_array[width_ends[~missing_widths].astype(int)] - \
                              time_array[width_starts[~missing_widths].astype(int)]
    if any(missing_widths):
        widths[missing_widths] = np.nan

    return widths

def _combine_and_interpolate(data):
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












