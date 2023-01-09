#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:49:27 2022

@author: julienballbe
"""




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle,geom_vline,geom_hline
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
from plotnine.labels import xlab
from plotnine.coords import coord_cartesian

import scipy
from scipy import signal
from scipy.stats import linregress
from scipy import optimize
from scipy.interpolate import splrep, splev
from scipy.misc import derivative
from scipy import io

import random

import os

from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel,GaussianModel
from lmfit import Parameters, Minimizer,fit_report

import sys
from sklearn.metrics import mean_squared_error

import logging
import Fit_library as fitlib
#%%



def identify_spike(membrane_trace_array,time_array,current_trace,stim_start_time,stim_end_time,do_plot=False):

    TPC_table=create_TPC(time_trace=time_array,
                         potential_trace=membrane_trace_array,
                         current_trace=current_trace,
                         filter=2.)

    time_derivative=TPC_table.loc[:,'Potential_first_time_derivative_mV/s']
    second_time_derivative=TPC_table.loc[:,'Potential_second_time_derivative_mV/s/s']
    
    preliminary_spike_index=detect_putative_spikes(v=membrane_trace_array,
                                                         t=time_array,
                                                         start=stim_start_time,
                                                         end=stim_end_time,
                                                         filter=2.,
                                                         dv_cutoff=20.,
                                                         dvdt=time_derivative)

    if do_plot:
        preliminary_spike_table=TPC_table.iloc[preliminary_spike_index[~np.isnan(preliminary_spike_index)],:]
        preliminary_spike_table['Feature']='A-Preliminary_spike_threshold'
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()+geom_point(preliminary_spike_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('1 - After detect_putative_spikes')
        print(current_plot)
        
    peak_index_array=find_peak_indexes(v=membrane_trace_array,
                                       t=time_array,
                                       spike_indexes=preliminary_spike_index,
                                       end=stim_end_time)

    if do_plot:
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:]

        peak_spike_table['Feature']='B-Preliminary_spike_peak'
        full_table=pd.concat([preliminary_spike_table,peak_spike_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()+geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('2 - After find_peak_index')
        print(current_plot)

    spike_threshold_index,peak_index_array=filter_putative_spikes(v=membrane_trace_array,
                                                                  t=time_array,
                                                                  spike_indexes=preliminary_spike_index,
                                                                  peak_indexes=peak_index_array,
                                                                  min_height=20.,
                                                                  min_peak=-30.,
                                                                  filter=2.,
                                                                  dvdt=time_derivative)

    if do_plot:
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        spike_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        
        full_table=pd.concat([spike_threshold_table,peak_spike_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('3 - After filter_putatyive_spikes')
        print(current_plot)
    
    upstroke_index=find_upstroke_indexes(v=membrane_trace_array,
                                         t=time_array,
                                         spike_indexes=spike_threshold_index,
                                         peak_indexes=peak_index_array,filter=2.,
                                         dvdt=time_derivative)

    if do_plot:
        upstroke_table=TPC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        full_table=pd.concat([full_table,upstroke_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('4 - After find_upstroke_indexes')
        print(current_plot)
        
        
    spike_threshold_index=refine_threshold_indexes(v=membrane_trace_array,
                                                   t=time_array,
                                                   peak_indexes=peak_index_array,
                                                   upstroke_indexes=upstroke_index,
                                                   thresh_frac=0.05,
                                                   filter=2.,
                                                   dvdt=time_derivative,
                                                   dv2dt2=second_time_derivative)

    if do_plot:
        refined_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; refined_threshold_table['Feature']='D-Refined_spike_threshold'
        full_table=pd.concat([full_table,refined_threshold_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('5 - After refine_threshold_indexes')
        print(current_plot)
        
    
    
    spike_threshold_index,peak_index_array,upstroke_index,clipped=check_thresholds_and_peaks(v=membrane_trace_array,
                                                                                            t=time_array,
                                                                                            spike_indexes=spike_threshold_index,
                                                                                            peak_indexes=peak_index_array,
                                                                                            upstroke_indexes=upstroke_index, 
                                                                                            start=stim_start_time, 
                                                                                            end=stim_end_time,
                                                                                            max_interval=0.01, 
                                                                                            thresh_frac=0.05, 
                                                                                            filter=2., 
                                                                                            dvdt=time_derivative,
                                                                                            tol=1.0, 
                                                                                            reject_at_stim_start_interval=0.)

    if do_plot:
        
        spike_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        upstroke_table=TPC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        
        full_table=pd.concat([spike_threshold_table,peak_spike_table,upstroke_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('6 - After check_thresholds_and_peaks')
        print(current_plot)
        
    trough_index=find_trough_indexes(v=membrane_trace_array,
                                     t=time_array,
                                     spike_indexes=spike_threshold_index,
                                     peak_indexes=peak_index_array,
                                     clipped=clipped,
                                     end=stim_end_time)

    if do_plot:

        trough_table=TPC_table.iloc[trough_index[~np.isnan(trough_index)].astype(int),:]; trough_table['Feature']='D-Trough'
        full_table=pd.concat([full_table,trough_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('7 - After find_trough_indexes')
        print(current_plot)
    
    

    fast_AHP_index=find_fast_AHP_indexes(v=membrane_trace_array,
                                     t=time_array,
                                     spike_indexes=spike_threshold_index,
                                     peak_indexes=peak_index_array, 
                                     clipped=clipped, 
                                     end=stim_end_time,
                                     dvdt=time_derivative)

    if do_plot:
        fast_AHP_table=TPC_table.iloc[fast_AHP_index[~np.isnan(fast_AHP_index)],:]; fast_AHP_table['Feature']='E-Fast_AHP'
        full_table=pd.concat([full_table,fast_AHP_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('8 - After find_fast_AHP_indexes')
        print(current_plot)
    
    
    
    # slow_trough_index=find_slow_trough_indexes(v=membrane_trace_array,
    #                                  t=time_array,
    #                                  spike_indexes=spike_threshold_index,
    #                                  fast_trough_indexes=fast_trough_index, 
    #                                  clipped=clipped, 
    #                                  end=stim_end_time)
    # if do_plot:
    #     slow_trough_table=TPC_table.iloc[slow_trough_index[~np.isnan(slow_trough_index)],:]; slow_trough_table['Feature']='F-Slow_Trough'
    #     full_table=pd.concat([full_table,slow_trough_table])
    #     current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
    #     current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
    #     current_plot+=xlim(stim_start_time,stim_end_time)
    #     current_plot+=xlab('9 - After find_slow_trough_indexes')
    #     print(current_plot)
    

    downstroke_index=find_downstroke_indexes(v=membrane_trace_array,
                                             t=time_array, 
                                             peak_indexes=peak_index_array,
                                             trough_indexes=trough_index,
                                             clipped=clipped,
                                             filter=2.,
                                             dvdt=time_derivative)

    if do_plot:
        downstroke_table=TPC_table.iloc[downstroke_index[~np.isnan(downstroke_index)],:]; downstroke_table['Feature']='F-Downstroke'
        full_table=pd.concat([full_table,downstroke_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('9 - After find_trough_indexes')
        print(current_plot)
        
        
    fast_trough_index, adp_index, slow_trough_index, clipped=find_fast_trough_adp_slow_trough(v=membrane_trace_array, 
                                                                                                    t=time_array, 
                                                                                                    spike_indexes=spike_threshold_index,
                                                                                                    peak_indexes=peak_index_array, 
                                                                                                    downstroke_indexes=downstroke_index,
                                                                                                    clipped=clipped,
                                                                                                    end=stim_end_time,
                                                                                                    filter=2.,
                                                                                                    heavy_filter=1.,
                                                                                                    downstroke_frac=.01,
                                                                                                    adp_thresh=1.5,
                                                                                                    tol=1.,
                                                                                                    flat_interval=.002,
                                                                                                    adp_max_delta_t=.005,
                                                                                                    adp_max_delta_v=10,
                                                                                                    dvdt=time_derivative)

    if do_plot:
        fast_trough_table=TPC_table.iloc[fast_trough_index[~np.isnan(fast_trough_index)],:]; fast_trough_table['Feature']='G-Fast_Trough'
        full_table=pd.concat([full_table,fast_trough_table])
        
        adp_table=TPC_table.iloc[adp_index[~np.isnan(adp_index)],:]; adp_table['Feature']='H-ADP'
        full_table=pd.concat([full_table,adp_table])
        
        slow_trough_table=TPC_table.iloc[slow_trough_index[~np.isnan(slow_trough_index)],:]; slow_trough_table['Feature']='I-Slow_Trough'
        full_table=pd.concat([full_table,slow_trough_table])
        
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('10 - After find_fast_trough/ADP/slow_trough_indexes')
        print(current_plot)
        
    

    spike_threshold_index = spike_threshold_index[~np.isnan(spike_threshold_index)].astype(int)
    peak_index_array = peak_index_array[~np.isnan(peak_index_array)].astype(int)
    upstroke_index = upstroke_index[~np.isnan(upstroke_index)].astype(int)
    downstroke_index = downstroke_index[~np.isnan(downstroke_index)].astype(int)
    trough_index = trough_index[~np.isnan(trough_index)].astype(int)
    fast_trough_index = fast_trough_index[~np.isnan(fast_trough_index)].astype(int)
    slow_trough_index = slow_trough_index[~np.isnan(slow_trough_index)].astype(int)
    adp_index = adp_index[~np.isnan(adp_index)].astype(int)
    fast_AHP_index = fast_AHP_index[~np.isnan(fast_AHP_index)].astype(int)

    
    
    
     
    
    
    spike_feature_dict={'Threshold':np.array(spike_threshold_index),
                        'Peak':np.array(peak_index_array),
                        'Upstroke':np.array(upstroke_index),
                        'Downstroke':np.array(downstroke_index),
                        'Trough':np.array(trough_index),
                        'Fast_Trough':np.array(fast_trough_index),
                        'Slow_Trough':np.array(slow_trough_index),
                        'ADP':np.array(adp_index),
                        'fAHP':np.array(fast_AHP_index)}
    
    return spike_feature_dict

    
def identify_spike_shorten(membrane_trace_array,time_array,current_trace,stim_start_time,stim_end_time,do_plot=False):

    TPC_table=create_TPC(time_trace=time_array,
                         potential_trace=membrane_trace_array,
                         current_trace=current_trace,
                         filter=2.)

    time_derivative=TPC_table.loc[:,'Potential_first_time_derivative_mV/s']
    second_time_derivative=TPC_table.loc[:,'Potential_second_time_derivative_mV/s/s']
    
    preliminary_spike_index=detect_putative_spikes(v=membrane_trace_array,
                                                         t=time_array,
                                                         start=stim_start_time,
                                                         end=stim_end_time,
                                                         filter=2.,
                                                         dv_cutoff=20.,
                                                         dvdt=time_derivative)

    if do_plot:
        preliminary_spike_table=TPC_table.iloc[preliminary_spike_index[~np.isnan(preliminary_spike_index)],:]
        preliminary_spike_table['Feature']='A-Preliminary_spike_threshold'
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()+geom_point(preliminary_spike_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('1 - After detect_putative_spikes')
        print(current_plot)
        
    peak_index_array=find_peak_indexes(v=membrane_trace_array,
                                       t=time_array,
                                       spike_indexes=preliminary_spike_index,
                                       end=stim_end_time)

    if do_plot:
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:]

        peak_spike_table['Feature']='B-Preliminary_spike_peak'
        full_table=pd.concat([preliminary_spike_table,peak_spike_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()+geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('2 - After find_peak_index')
        print(current_plot)

    spike_threshold_index,peak_index_array=filter_putative_spikes(v=membrane_trace_array,
                                                                  t=time_array,
                                                                  spike_indexes=preliminary_spike_index,
                                                                  peak_indexes=peak_index_array,
                                                                  min_height=20.,
                                                                  min_peak=-30.,
                                                                  filter=2.,
                                                                  dvdt=time_derivative)

    if do_plot:
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        spike_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        
        full_table=pd.concat([spike_threshold_table,peak_spike_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('3 - After filter_putatyive_spikes')
        print(current_plot)
    
    upstroke_index=find_upstroke_indexes(v=membrane_trace_array,
                                         t=time_array,
                                         spike_indexes=spike_threshold_index,
                                         peak_indexes=peak_index_array,filter=2.,
                                         dvdt=time_derivative)

    if do_plot:
        upstroke_table=TPC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        full_table=pd.concat([full_table,upstroke_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('4 - After find_upstroke_indexes')
        print(current_plot)
        
        
    spike_threshold_index=refine_threshold_indexes(v=membrane_trace_array,
                                                   t=time_array,
                                                   peak_indexes=peak_index_array,
                                                   upstroke_indexes=upstroke_index,
                                                   thresh_frac=0.05,
                                                   filter=2.,
                                                   dvdt=time_derivative,
                                                   dv2dt2=second_time_derivative)

    if do_plot:
        refined_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; refined_threshold_table['Feature']='D-Refined_spike_threshold'
        full_table=pd.concat([full_table,refined_threshold_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('5 - After refine_threshold_indexes')
        print(current_plot)
        
    
    
    spike_threshold_index,peak_index_array,upstroke_index,clipped=check_thresholds_and_peaks(v=membrane_trace_array,
                                                                                            t=time_array,
                                                                                            spike_indexes=spike_threshold_index,
                                                                                            peak_indexes=peak_index_array,
                                                                                            upstroke_indexes=upstroke_index, 
                                                                                            start=stim_start_time, 
                                                                                            end=stim_end_time,
                                                                                            max_interval=0.01, 
                                                                                            thresh_frac=0.05, 
                                                                                            filter=2., 
                                                                                            dvdt=time_derivative,
                                                                                            tol=1.0, 
                                                                                            reject_at_stim_start_interval=0.)

    if do_plot:
        
        spike_threshold_table=TPC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        peak_spike_table=TPC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        upstroke_table=TPC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        
        full_table=pd.concat([spike_threshold_table,peak_spike_table,upstroke_table])
        current_plot=ggplot(TPC_table,aes(x='Time_s',y="Membrane_potential_mV"))+geom_line()
        current_plot+=geom_point(full_table,aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=xlim(stim_start_time,stim_end_time)
        current_plot+=xlab('6 - After check_thresholds_and_peaks')
        print(current_plot)
        
    
    

    
    if len(spike_threshold_index)>0:
        return True
    else:
        return False
    
    
def create_TPC(time_trace,potential_trace,current_trace,filter=None):
    '''
    Create table containing Time, Potential, Current, First and Second time derivative of membrane potential

    Parameters
    ----------
    time_trace : np.array
        Time points array in s.
    potential_trace : np.array
        Array of membrane potential recording in mV.
    current_trace : np.array
        Array of input current in pA.
    filter: cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Raises
    ------
    ValueError
        All array must be of the amse length, if not raises ValueError.

    Returns
    -------
    TPC_table : DataFrame
        DataFrame with 5 columns
            'Time_s': Time in s
            'Membrane_potential_mV': Membrane potential in mV
            'Input_current_pA': input current in pA
            'Potential_first_time_derivative_mV/s' : First time derivative of the membrane potential in mV/s
            'Potential_second_time_derivative_mV/s/s' : Second time derivative of the membrane potential in mV/s/s

    '''
    length=len(time_trace)
    
    if length!=len(potential_trace) or length!=len(current_trace):

        raise ValueError('All lists must be of equal length. Current lengths: Time {}, Potential {}, Current {}'.format(len(time_trace),len(potential_trace),len(current_trace)))
    
    first_derivative=calculate_dvdt(potential_trace,time_trace,filter=filter)
    first_derivative=np.insert(first_derivative,0,np.nan)

    second_derivative=calculate_dvdt(first_derivative,time_trace,filter=None)
    second_derivative=np.insert(second_derivative,0,[np.nan,np.nan])
    
    TPC_table=pd.DataFrame({'Time_s':time_trace,
                               'Membrane_potential_mV':potential_trace,
                               'Input_current_pA':current_trace,
                               'Potential_first_time_derivative_mV/s':first_derivative,
                               'Potential_second_time_derivative_mV/s/s':second_derivative},
                           dtype=np.float64) 
    
    return TPC_table


    
    
    
def get_smooth_deriv(membrane_potential_table,do_plot=False):
    '''
    

    Parameters
    ----------
    membrane_potential_table : DataFrame
        First column represent time in s
        Second column represent membran potential in mV
        
    do_plot : Boolean, optional
        If true, print the plot

    Returns
    -------
    derivative_table : DataFrame
        First column represents time in s
        Second column represents membrane potential derivative in mV/ms.

    '''
    
    
    derivative_trace=pd.Series(calculate_dvdt(v=membrane_potential_table.iloc[:,1],
                                    t=membrane_potential_table.iloc[:,0],
                                    filter=None))
    derivative_table=pd.DataFrame(np.column_stack((membrane_potential_table.iloc[1:,0],derivative_trace)),columns=['Time_s',"Membrane_potential_derivative_mV/ms"])
    
    if do_plot:
        myplot=ggplot(derivative_table,aes(x=derivative_table.iloc[:,0],y=derivative_table.iloc[:,1]))+geom_line(color='red')+coord_cartesian(xlim=(0.05,0.55))
        
        myplot
        
    membrane_potential_table=pd.merge(membrane_potential_table,derivative_table,on='Time_s',how='outer')
    return membrane_potential_table


def has_fixed_dt(t): 
    """Check that all time intervals are identical."""
    dt = np.diff(t)
    
    return np.allclose(dt, np.ones_like(dt) * dt[0])

def calculate_dvdt(v, t, filter=None): 
    """Low-pass filters (if requested) and differentiates voltage by time.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """

    if has_fixed_dt(t) and filter:
        delta_t = t[1] - t[0]
        sample_freq = 1. / delta_t
        filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency
        if filt_coeff < 0 or filt_coeff >= 1:
            raise ValueError("bessel coeff ({:f}) is outside of valid range [0,1); cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
        b, a = signal.bessel(4, filt_coeff, "low")
        v_filt = signal.filtfilt(b, a, v, axis=0)
        dv = np.diff(v_filt)
    else:
        dv = np.diff(v)

    dt = np.diff(t)
    dvdt = 1e-3 * dv / dt # in V/s = mV/ms

    # Remove nan values (in case any dt values == 0)
    dvdt = dvdt[~np.isnan(dvdt)]

    return dvdt

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

def spike_detect_calculate_dvdt(v, t, filter=None):
    """Low-pass filters (if requested) and differentiates voltage by time.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """

    if has_fixed_dt(t) and filter:
        delta_t = t[1] - t[0]
        sample_freq = 1. / delta_t
        filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency
        if filt_coeff < 0 or filt_coeff >= 1:
            raise ValueError("bessel coeff ({:f}) is outside of valid range [0,1); cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
        b, a = signal.bessel(4, filt_coeff, "low")
        v_filt = signal.filtfilt(b, a, v, axis=0)
        dv = np.diff(v_filt)
    else:
        dv = np.diff(v)

    dt = np.diff(t)
    dvdt = 1e-3 * dv / dt  # in V/s = mV/ms

    # some data sources, such as neuron, occasionally report 
    # duplicate timestamps, so we require that dt is not 0
    # np.fabs(dt) --> absolute values of dt
    return dvdt[np.fabs(dt) > sys.float_info.epsilon]


def detect_putative_spikes(v, t, start=None, end=None, filter=10., dv_cutoff=20., dvdt=None):
    """Perform initial detection of spikes and return their indexes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional)
    end : end of time window for spike detection (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    putative_spikes : numpy array of preliminary spike indexes
    """

    if not isinstance(v, np.ndarray):
        raise TypeError("v is not an np.ndarray")

    if not isinstance(t, np.ndarray):
        raise TypeError("t is not an np.ndarray")

    # if v.shape != t.shape:
    #     raise er.FeatureError("Voltage and time series do not have the same dimensions")

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, end)
    v_window = v[start_index:end_index + 1]
    t_window = t[start_index:end_index + 1]

    if dvdt is None:
        dvdt =spike_detect_calculate_dvdt(v_window, t_window, filter)
    else:
        dvdt = dvdt[start_index:end_index]

    # Find positive-going crossings of dV/dt cutoff level
    

    putative_spikes = np.flatnonzero(np.diff(np.greater_equal(dvdt, dv_cutoff).astype(int)) == 1)

    if len(putative_spikes) <= 1:
        # Set back to original index space (not just window)
        return np.array(putative_spikes) + start_index

    # Only keep spike times if dV/dt has dropped all the way to zero between putative spikes
    putative_spikes = [putative_spikes[0]] + [s for i, s in enumerate(putative_spikes[1:])
        if np.any(dvdt[putative_spikes[i]:s] < 0)]

    # Set back to original index space (not just window)
    return np.array(putative_spikes) + start_index


def find_peak_indexes(v, t, spike_indexes, end=None):
    """Find indexes of spike peaks.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    end : end of time window for spike detection (optional)
    """

    if not end:
        end = t[-1]
    end_index = find_time_index(t, end)

    spks_and_end = np.append(spike_indexes, end_index)
    peak_indexes = [np.argmax(v[spk:next]) + spk for spk, next in
                    zip(spks_and_end[:-1], spks_and_end[1:])]
    #finds index of maximum value between two consecutive spikes index
    #spk represents the first spike index (going through the list of spike_index without the last one)
    #next respesent the second spike index (going through the list of spike index without the first one)
    #np.argmax(v[spk:next]) + spk --> add spk because is initilized at 0 in (v[spk:next])

    return np.array(peak_indexes)

def filter_putative_spikes(v, t, spike_indexes, peak_indexes, min_height=2.,
                           min_peak=-30., filter=10., dvdt=None):
    """Filter out events that are unlikely to be spikes based on:
        * Height (threshold to peak)
        * Absolute peak level

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
    min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    spike_indexes : numpy array of threshold indexes
    peak_indexes : numpy array of peak indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([]), np.array([])

    if dvdt is None:
        dvdt = spike_detect_calculate_dvdt(v, t, filter)

    diff_mask = [np.any(dvdt[peak_ind:spike_ind] < 0)
                 for peak_ind, spike_ind
                 in zip(peak_indexes[:-1], spike_indexes[1:])]
    #diif_mask --> check if the derivative between a peak and the next threshold ever goes negative
    
    peak_indexes = peak_indexes[np.array(diff_mask + [True])]
    spike_indexes = spike_indexes[np.array([True] + diff_mask)]
    #keep only peak indexes where diff mask was True (same for spike index)

    peak_level_mask = v[peak_indexes] >= min_peak
    #check if identified peaks are higher than minimum peak values (defined)
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]
    #keep only spike and peaks if spike_peak is higher than minimum value

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= min_height
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]

    #keep only events where the potential difference between peak and threshold is higher than minimum height (defined)
    return spike_indexes, peak_indexes

def find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=10., dvdt=None):
    """Find indexes of maximum upstroke of spike.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    upstroke_indexes : numpy array of upstroke indexes

    """

    if dvdt is None:
        dvdt = spike_detect_calculate_dvdt(v, t, filter)

    upstroke_indexes = [np.argmax(dvdt[spike:peak]) + spike for spike, peak in
                        zip(spike_indexes, peak_indexes)]

    return np.array(upstroke_indexes)


def refine_threshold_indexes(v, t, peak_indexes,upstroke_indexes, thresh_frac=0.05, filter=10., dvdt=None,dv2dt2=None):
    """Refine threshold detection of previously-found spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    upstroke_indexes : numpy array of indexes of spike upstrokes (for threshold target calculation)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    threshold_indexes : numpy array of threshold indexes
    """

    if not peak_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = spike_detect_calculate_dvdt(v, t, filter)
    
    ##########
    ## Here the threshold is defined as the local maximum of the second potential derivative
    # opening_window=np.append(np.array([0]), peak_indexes[:-1])
    # closing_window=peak_indexes
    
    # threshold_indexes = [np.argmax(dv2dt2[prev_peak:nex_peak])+prev_peak for prev_peak, nex_peak in
    #                     zip(opening_window, closing_window)]
    # return np.array(threshold_indexes)
    ##########
    
    ## Here the threshold is defined as the last index where dvdt= avg_upstroke * thresh_frac
    avg_upstroke = dvdt[upstroke_indexes].mean()
    target = avg_upstroke * thresh_frac

    upstrokes_and_start = np.append(np.array([0]), upstroke_indexes)
    threshold_indexes = []
    for upstk, upstk_prev in zip(upstrokes_and_start[1:], upstrokes_and_start[:-1]):

        potential_indexes = np.flatnonzero(dvdt[upstk:upstk_prev:-1] <= target)

        if not potential_indexes.size:
            # couldn't find a matching value for threshold,
            # so just going to the start of the search interval
            threshold_indexes.append(upstk_prev)
        else:
            threshold_indexes.append(upstk - potential_indexes[0])

    return np.array(threshold_indexes)

def check_thresholds_and_peaks(v, t, spike_indexes, peak_indexes, upstroke_indexes, start=None, end=None,
                               max_interval=0.01, thresh_frac=0.05, filter=10., dvdt=None,
                               tol=1.0, reject_at_stim_start_interval=0.):
    """Validate thresholds and peaks for set of spikes

    Check that peaks and thresholds for consecutive spikes do not overlap
    Spikes with overlapping thresholds and peaks will be merged.

    Check that peaks and thresholds for a given spike are not too far apart.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    upstroke_indexes : numpy array of indexes of spike upstrokes
    start : start of time window for feature analysis (optional)
    end : end of time window for feature analysis (optional)
    max_interval : maximum allowed time between start of spike and time of peak in sec (default 0.005)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)
    tol : tolerance for returning to threshold in mV (optional, default 1)
    reject_at_stim_start_interval : duration of window after start to reject potential spikes (optional, default 0)

    Returns
    -------
    spike_indexes : numpy array of modified spike indexes
    peak_indexes : numpy array of modified spike peak indexes
    upstroke_indexes : numpy array of modified spike upstroke indexes
    clipped : numpy array of clipped status of spikes
    """

    if start is not None and reject_at_stim_start_interval > 0:
        mask = t[spike_indexes] > (start + reject_at_stim_start_interval)
        spike_indexes = spike_indexes[mask]
        peak_indexes = peak_indexes[mask]
        upstroke_indexes = upstroke_indexes[mask]
    
    overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1] + 1)
    
    if overlaps.size:
        spike_mask = np.ones_like(spike_indexes, dtype=bool)
        spike_mask[overlaps + 1] = False
        spike_indexes = spike_indexes[spike_mask]

        peak_mask = np.ones_like(peak_indexes, dtype=bool)
        peak_mask[overlaps] = False
        peak_indexes = peak_indexes[peak_mask]

        upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
        upstroke_mask[overlaps] = False
        upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []

    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):

        if t[peak] - t[spk] >= max_interval:
            logging.info("Need to recalculate threshold-peak pair that exceeds maximum allowed interval ({:f} s)".format(max_interval))
            too_long_spikes.append(i)

    if too_long_spikes:
        if dvdt is None:
            dvdt = spike_detect_calculate_dvdt(v, t, filter)
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * thresh_frac

        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = find_time_index(t, t[peak] - max_interval)
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            
            if not below_target.size:

                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]

                t_0 = find_time_index(t, t[spike] + 2 * max_interval)
                new_peak = np.argmax(v[spike:t_0]) + spike
               
                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                
                if t[new_peak] - t[spike] < max_interval and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak

                else:
                    # Otherwise, log and get rid of the spike
                    logging.info("Could not redetermine threshold-peak pair - dropping that pair")

                    drop_spikes.append(i)

#                     raise FeatureError("Could not redetermine threshold")
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]

        if drop_spikes:

            spike_indexes = np.delete(spike_indexes, drop_spikes)
            peak_indexes = np.delete(peak_indexes, drop_spikes)
            upstroke_indexes = np.delete(upstroke_indexes, drop_spikes)

    if not end:
        end = t[-1]
    end_index = find_time_index(t, end)

    clipped = find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol)

    return spike_indexes, peak_indexes, upstroke_indexes, clipped

def find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol):
    """
    Check that last spike was not cut off too early by end of stimulus
    by checking that the membrane potential returned to at least the threshold
    voltage - otherwise, drop it

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    end_index: int index of the end of time window for feature analysis

    tol: float tolerance to returning to threshold

    Returns
    -------
    clipped: Boolean np.array
    """
    clipped = np.zeros_like(spike_indexes, dtype=bool)

    if len(spike_indexes)>0:
        vtail = v[peak_indexes[-1]:end_index + 1]
        if not np.any(vtail <= v[spike_indexes[-1]] + tol):
           
            logging.debug(
                "Failed to return to threshold voltage + tolerance (%.2f) after last spike (min %.2f) - marking last spike as clipped",
                v[spike_indexes[-1]] + tol, vtail.min())
            clipped[-1] = True
            logging.debug("max %f, min %f, t(end_index):%f" % (np.max(vtail), np.min(vtail), t[end_index]))

    return clipped


def find_fast_AHP_indexes(v, t, spike_indexes, peak_indexes, clipped=None, end=None,dvdt=None):
    """
    Detect the minimum membrane potential value in the 5ms time window after a spike peak. 
    For a given spike, fAHP is defined if during the 5ms following the peak:
        -	The time derivative of the membrane potential went positive
        and
        -   The membrane potential went below the spike threshold
    If either if this condition was not observed, fAHP was not defined for this spike



    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)

    Returns
    -------
    fast_AHP_indexes : numpy array of threshold indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)
    
    if dvdt is None:
        dvdt = spike_detect_calculate_dvdt(v, t, filter)
    
    
    window_closing_time=t[peak_indexes]+.005
    window_closing_idx=[]
    for current_time in window_closing_time:
        if current_time <= t[-1]:
            window_closing_idx.append(find_time_index(t, current_time))
        else:
            window_closing_idx.append(find_time_index(t, t[-1]))
    #window_closing_idx=find_time_index(t, window_closing_time)
    
    
    fast_AHP_indexes = np.zeros_like(window_closing_idx, dtype=float)
    # trough_indexes[:-1] = [v[peak:spk].argmin() + peak for peak, spk
    #                        in zip(peak_indexes[:-1], spike_indexes[1:])]
    fast_AHP_indexes=[]
    for peak , wnd, thresh in zip(peak_indexes, window_closing_idx,spike_indexes):
        if np.flatnonzero(dvdt[peak:wnd] >= 0).size and min(v[peak:wnd])<=thresh :
            fast_AHP_indexes.append(int(v[peak:wnd].argmin() + peak ))
            
        
    # fast_AHP_indexes = [v[peak:wnd].argmin() + peak if np.flatnonzero(dvdt[peak:wnd] >= 0) and
    #                     np.min(v[peak:wnd])<=thresh 
    #                     for peak , wnd,thresh
    #                         in zip(peak_indexes, window_closing_idx,spike_indexes) ] #Detect lowest voltage value in a 5ms window after last peak
    
    if fast_AHP_indexes[-1]>=end_index: #If the last fast trough detected is after end_time index (because of the closing window of 5ms), redefine it at end_index
        fast_AHP_indexes[-1]=end_index
    # if clipped[-1]:
    #     # If last spike is cut off by the end of the window, trough is undefined
    #     trough_indexes[-1] = np.nan
    # else:
    #     trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method
    
    # # If peak is the same point as the trough, drop that point
    # trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]

    return np.array(fast_AHP_indexes)

def find_slow_trough_indexes(v, t, spike_indexes, fast_trough_indexes, clipped=None, end=None):
    """
    Find indexes of minimum voltage (trough) between spikes.
    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    Returns
    -------
    slow_trough_indexes : numpy array of threshold indexes
    """
    
    if not spike_indexes.size or not fast_trough_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)

    slow_trough_indexes = np.zeros_like(spike_indexes, dtype=float)
    slow_trough_indexes[:-1] = [v[fast_trough:spk].argmin() + fast_trough for fast_trough, spk
                           in zip(fast_trough_indexes[:-1], spike_indexes[1:])]
    
    if clipped[-1]:
        # If last spike is cut off by the end of the window, trough is undefined
        slow_trough_indexes[-1] = np.nan
    else:
        if fast_trough_indexes[-1] == end_index:
            slow_trough_indexes[-1]=fast_trough_indexes[-1]
            
        else:
            
            slow_trough_indexes[-1] = v[fast_trough_indexes[-1]:end_index].argmin() + fast_trough_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method

    return np.array(slow_trough_indexes)

def find_trough_indexes(v, t, spike_indexes, peak_indexes, clipped=None, end=None):
    """
    Find indexes of minimum voltage (trough) between spikes.
    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    Returns
    -------
    trough_indexes : numpy array of threshold indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)

    trough_indexes = np.zeros_like(spike_indexes, dtype=float)
    trough_indexes[:-1] = [int(v[peak:spk].argmin() + peak) for peak, spk
                           in zip(peak_indexes[:-1], spike_indexes[1:])]
    
    if clipped[-1]:
        # If last spike is cut off by the end of the window, trough is undefined
        trough_indexes[-1] = np.nan
    else:
        trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method

    # If peak is the same point as the trough, drop that point

    trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]

    
    return np.array(trough_indexes)

def find_downstroke_indexes(v, t, peak_indexes, trough_indexes, clipped=None, filter=10., dvdt=None):
    """Find indexes of minimum voltage (troughs) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of threshold indexes
    clipped: boolean array - False if spike not clipped by edge of window
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    downstroke_indexes : numpy array of downstroke indexes
    """

    if not trough_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = spike_detect_calculate_dvdt(v, t, filter)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes, dtype=bool)

    if len(peak_indexes) < len(trough_indexes):
        raise ValueError("Cannot have more troughs than peaks")
# Taking this out...with clipped info, should always have the same number of points
 #   peak_indexes = peak_indexes[:len(trough_indexes)]

    
    valid_peak_indexes = peak_indexes[~clipped].astype(int)
    
    valid_trough_indexes = trough_indexes[~clipped].astype(int)

    downstroke_indexes = np.zeros_like(peak_indexes) * np.nan


    downstroke_index_values = [np.argmin(dvdt[peak:trough]) + peak for peak, trough
                         in zip(valid_peak_indexes, valid_trough_indexes)]

    downstroke_indexes[~clipped] = downstroke_index_values

    return downstroke_indexes

def find_fast_trough_adp_slow_trough(v, t, spike_indexes, peak_indexes, downstroke_indexes, clipped=None, end=None, filter=10.,
                           heavy_filter=1., downstroke_frac=0.01, adp_thresh=1.5, tol=1.,
                           flat_interval=0.002, adp_max_delta_t=0.005, adp_max_delta_v=10., dvdt=None):
    """
    Comes from analyze_trough_details in IPFX
    Analyze trough to determine if an ADP exists and whether the reset is a 'detour' or 'direct'

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike threshold indexes
    peak_indexes : numpy array of spike peak indexes
    downstroke_indexes : numpy array of spike downstroke indexes
    end : end of time window (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    heavy_filter : lower cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    downstroke_frac : fraction of spike downstroke to define spike fast trough (optional, default 0.01)
    adp_thresh: minimum dV/dt in V/s to exceed to be considered to have an ADP (optional, default 1.5)
    tol : tolerance for evaluating whether Vm drops appreciably further after end of spike (default 1.0 mV)
    flat_interval: if the trace is flat for this duration, stop looking for an ADP (default 0.002 s)
    adp_max_delta_t: max possible ADP delta t (default 0.005 s)
    adp_max_delta_v: max possible ADP delta v (default 10 mV)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    isi_types : numpy array of isi reset types (direct or detour)
    fast_trough_indexes : numpy array of indexes at the start of the trough (i.e. end of the spike)
    adp_indexes : numpy array of adp indexes (np.nan if there was no ADP in that ISI
    slow_trough_indexes : numpy array of indexes at the minimum of the slow phase of the trough
                          (if there wasn't just a fast phase)
    """

    if end is None:
        end = t[-1]
    end_index = find_time_index(t, end)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes)

    # Can't evaluate for spikes that are clipped by the window
    orig_len = len(peak_indexes)

    valid_spike_indexes = spike_indexes[~clipped]
    valid_peak_indexes = peak_indexes[~clipped]

    if dvdt is None:
        dvdt = calculate_dvdt(v, t, filter)

    dvdt_hvy = calculate_dvdt(v, t, heavy_filter)

    # Writing as for loop - see if I can vectorize any later
    fast_trough_indexes = []
    adp_indexes = []
    slow_trough_indexes = []
    isi_types = []

    update_clipped = []

    for dwnstk, next_spk in zip(downstroke_indexes,np.append(valid_spike_indexes[1:], end_index)):
        target = downstroke_frac * dvdt[dwnstk]
        dwnstk=int(dwnstk)
        terminated_points = np.flatnonzero(dvdt[dwnstk:next_spk] >= target)
        #terminated points corresponds  between spike downstroke  and next spike threshold to where dvdt gets higher than 1% of dvdt at spike dowstroke 
        # After the downsteroke the dvdt is neg, so 1% of dvdt at downstroke corresponds to a flatening of the membrane potential
        if terminated_points.size:
            terminated = terminated_points[0] + dwnstk
            update_clipped.append(False)
        else:
            logging.debug("Could not identify fast trough - marking spike as clipped")
            isi_types.append(np.nan)
            fast_trough_indexes.append(np.nan)
            adp_indexes.append(np.nan)
            slow_trough_indexes.append(np.nan)
            update_clipped.append(True)
            continue
        
    
        # Could there be an ADP?
        adp_index = np.nan
        dv_over_thresh = np.flatnonzero(dvdt_hvy[terminated:next_spk] >= adp_thresh) # did the dvdt went higher than adp threshold after the fist index where dvdt went higher thant 0.01*dvdt at downstroke
        if dv_over_thresh.size:
            cross = dv_over_thresh[0] + terminated #cross = first index after downstroke that crosses the derivative threshold to look for ADP

            # only want to look for ADP before things get pretty flat
            # otherwise, could just pick up random transients long after the spike
            if t[cross] - t[terminated] < flat_interval:
                # Going back up fast, but could just be going into another spike
                # so need to check for a reversal (zero-crossing) in dV/dt
                zero_return_vals = np.flatnonzero(dvdt_hvy[cross:next_spk] <= 0) #zero_return_vals = where dvdt is negative between cross and next spike
                if zero_return_vals.size:
                    putative_adp_index = zero_return_vals[0] + cross #putative_adp_index = first index of dvdt positive after cross
                    min_index = v[putative_adp_index:next_spk].argmin() + putative_adp_index
                    if (v[putative_adp_index] - v[min_index] >= tol and # potential difference between min and first index of dvdt positive after cross must be higher than tol
                            v[putative_adp_index] - v[terminated] <= adp_max_delta_v and # potential difference between fats trough and first index of dvdt positive after cross, must be lower than max adp tolerated voltage
                            t[putative_adp_index] - t[terminated] <= adp_max_delta_t):
                        adp_index = putative_adp_index
                        slow_phase_min_index = min_index
                        isi_type = "detour"

        if np.isnan(adp_index):
            v_term = v[terminated]
            min_index = v[terminated:next_spk].argmin() + terminated
            if v_term - v[min_index] >= tol:
                # dropped further after end of spike -> detour reset
                isi_type = "detour"
                slow_phase_min_index = min_index
            else:
                isi_type = "direct"

        isi_types.append(isi_type)
        fast_trough_indexes.append(terminated)
        adp_indexes.append(adp_index)
        if isi_type == "detour":
            slow_trough_indexes.append(slow_phase_min_index)
        else:
            slow_trough_indexes.append(np.nan)
    
    clipped[~clipped] = update_clipped
    
    return np.array(fast_trough_indexes), np.array(adp_indexes), np.array(slow_trough_indexes), clipped
    # # If we had to kick some spikes out before, need to add nans at the end
    # output = []
    # output.append(np.array(isi_types))
    # for d in (fast_trough_indexes, adp_indexes, slow_trough_indexes):
    #     output.append(np.array(d, dtype=float))

    # if orig_len > len(isi_types):
    #     extra = np.zeros(orig_len - len(isi_types)) * np.nan
    #     output = tuple((np.append(o, extra) for o in output))

    # # The ADP and slow trough for the last spike in a train are not reliably
    # # calculated, and usually extreme when wrong, so we will NaN them out.
    # #
    # # Note that this will result in a 0 value when delta V or delta T is
    # # calculated, which may not be strictly accurate to the trace, but the
    # # magnitude of the difference will be less than in many of the erroneous
    # # cases seen otherwise

    # output[2][-1] = np.nan # ADP
    # output[3][-1] = np.nan # slow trough

    # clipped[~clipped] = update_clipped
    # return output, clipped

def plot_trace(TPC_table,feature_table,potential_derivative,stim_start,stim_end):
    potential_trace=["Membrane_potential_mV","Potential_first_time_derivative_mV/s","Potential_second_time_derivative_mV/s/s"]
    trace_to_plot=potential_trace[potential_derivative]
    my_plot=ggplot(TPC_table,aes(x="Time_s",y=str(trace_to_plot)))+geom_line()
    my_plot+=geom_point(feature_table,aes(x='Time_s',y=str(trace_to_plot),color='Feature'))
    my_plot+=xlim(stim_start,stim_end)
    
    return my_plot

def compute_input_resistance_time_cst(original_TPC_table,original_SF_table):
    TPC_table=original_TPC_table.copy()
    SF_table=original_SF_table.copy()
    sweep_id_list=np.array(TPC_table.loc[:,'Sweep'])
    TPC_table['Membrane_baseline_mV']=np.nan
    TPC_table['Membrane_SS_mV']=np.nan
    TPC_table['Stimulus_baseline_pA']=np.nan
    TPC_table['Stimulus_amplitude_pA']=np.nan
    TPC_table['Time_cst_ms']=np.nan
    for current_sweep in sweep_id_list:
        current_SF=pd.DataFrame(SF_table.loc[current_sweep,'SF']).copy()
        if current_SF.shape[0]==0:
            current_TPC=pd.DataFrame(TPC_table.loc[current_sweep,'TPC']).copy()
            membrane_trace=current_TPC.loc[:,'Membrane_potential_mV']
            time_trace=current_TPC.loc[:,'Time_s']
            stim_trace=current_TPC.loc[:,'Input_current_pA']
            stim_start_time=SF_table.loc[current_sweep,'Stim_start_s']
            stim_end_time=SF_table.loc[current_sweep,'Stim_end_s']
            stim_amplitude=SF_table.loc[current_sweep,'Stim_amp_pA']
            start_idx = np.argmin(abs(time_trace - stim_start_time))
            end_idx = np.argmin(abs(time_trace - stim_end_time))
            
            membrane_baseline=np.mean(membrane_trace[:start_idx])
            
            best_A,best_tau,membrane_SS=fitlib.fit_membrane_trace(current_TPC,stim_start_time,stim_end_time,do_plot=False)
            TPC_table.loc[current_sweep,'Membrane_baseline_mV']=membrane_baseline
            TPC_table.loc[current_sweep,'Membrane_SS_mV']=membrane_SS
            TPC_table.loc[current_sweep,'Stimulus_baseline_pA']=np.mean(stim_trace[:start_idx])
            TPC_table.loc[current_sweep,'Stimulus_amplitude_pA']=stim_amplitude
            TPC_table.loc[current_sweep,'Time_cst_ms']=best_tau*1e3 #convert s to ms
    
    #return TPC_table
    Input_resistance=np.mean((TPC_table.loc[:,"Membrane_SS_mV"]-TPC_table.loc[:,"Membrane_baseline_mV"])/(TPC_table.loc[:,"Stimulus_amplitude_pA"]-TPC_table.loc[:,"Stimulus_baseline_pA"]))*1e3 #Convert to MOhms
    IR_sd=np.std((TPC_table.loc[:,"Membrane_SS_mV"]-TPC_table.loc[:,"Membrane_baseline_mV"])/(TPC_table.loc[:,"Stimulus_amplitude_pA"]-TPC_table.loc[:,"Stimulus_baseline_pA"]))*1e3 #Convert to MOhms
    
    Time_cst=np.mean(TPC_table.loc[:,"Time_cst_ms"])
    Time_cst_sd=np.std(TPC_table.loc[:,'Time_cst_ms'])
    return Input_resistance,IR_sd,Time_cst,Time_cst_sd


def estimate_bridge_error(original_TPC_table,stim_amplitude,stim_start_time,stim_end_time,do_plot=False):
    
    
    TPC_table=original_TPC_table.reset_index(drop=True).copy()
    point_table=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Feature"])
    line_table=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Data"])
    # line_table=original_TPC_table.reset_index(drop=True).copy()
    # line_table=line_table.loc[:,["Time_s","Membrane_potential_mV"]]
    # line_table["Data"]='Original_data'
    start_time_index = np.argmin(abs(np.array(TPC_table['Time_s']) - stim_start_time))
    end_time_index = np.argmin(abs(np.array(TPC_table['Time_s']) - stim_end_time))
    stimulus_baseline=np.mean(TPC_table.loc[:(start_time_index-1),'Input_current_pA'])
    at_least_one_Bridge_Error=False
    
    ########################################################################################
    #Stimulus_start_window
    
    #Consider a 40msec windows centered on the specified stimulus start so +/- 20 msec and compute spline fit for current and potential
    stimulus_start_table=TPC_table[TPC_table["Time_s"]<(stim_start_time+.020)]
    stimulus_start_table=stimulus_start_table[stimulus_start_table['Time_s']>(stim_start_time-.020)]
   
    
    time_array=np.array(stimulus_start_table.loc[:,'Time_s'])
    
    
    current_start_trace_spline=scipy.interpolate.UnivariateSpline(np.array(stimulus_start_table.loc[:,'Time_s']),np.array(stimulus_start_table.loc[:,'Input_current_pA']))
    current_start_trace_spline.set_smoothing_factor(.999)
    current_trace_filtered=current_start_trace_spline(time_array)
    current_trace_filtered_derivative=calculate_dvdt ( current_trace_filtered , time_array )
    current_trace_filtered_derivative=np.insert(current_trace_filtered_derivative,0,current_trace_filtered_derivative[0])
    
    
    stimulus_start_table['Input_current_derivative_pA/s']=current_trace_filtered_derivative
    
    
    #determine T_trans
    T_trans_table=stimulus_start_table[stimulus_start_table["Time_s"]<(stim_start_time+.002)]
    T_trans_table=T_trans_table[T_trans_table['Time_s']>(stim_start_time-.002)]
    
    T_Trans_current_derivative=np.array(T_trans_table['Input_current_derivative_pA/s'])
    
    if stim_amplitude <= stimulus_baseline:
        
        max_abs_dI_dt_index = np.nanargmin(T_Trans_current_derivative)
        
        
    elif stim_amplitude>stimulus_baseline:

        max_abs_dI_dt_index = np.nanargmax(T_Trans_current_derivative)
        
        

    T_trans=np.array(T_trans_table.loc[:,'Time_s'])[max_abs_dI_dt_index]
    
    
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans-0.006 and T_trans-0.002 (and vice-versa)
    Pre_Trans_Median_table=stimulus_start_table[stimulus_start_table['Time_s']<=T_trans-.002]
    Pre_Trans_Median_table=Pre_Trans_Median_table[Pre_Trans_Median_table['Time_s']>=T_trans-.006]
    Pre_Trans_Median=np.median(Pre_Trans_Median_table['Input_current_pA'])
    
    Post_Trans_Median_table=stimulus_start_table[stimulus_start_table['Time_s']>=T_trans+.002]
    Post_Trans_Median_table=Post_Trans_Median_table[Post_Trans_Median_table['Time_s']<=T_trans+.006]
    Post_Trans_Median=np.median(Post_Trans_Median_table['Input_current_pA'])
    
    I_step=Post_Trans_Median-Pre_Trans_Median
    
    #determine DT_trans, the time scale of response transient to avoid contaminated period

    ## get time of the max (min) dV/dt, for a positive (negative) Istep, respectively
    DT_Trans_table=stimulus_start_table[stimulus_start_table['Time_s']>=T_trans].reset_index(drop=True).copy()


    DT_Trans_table=DT_Trans_table[DT_Trans_table['Time_s']<=(T_trans+.003)]
    potential_derivative_trace=np.array(DT_Trans_table.loc[:,'Potential_first_time_derivative_mV/s'])
    
    if stim_amplitude <= stimulus_baseline:

        min_dV_dt_index=np.nanargmin(potential_derivative_trace)
        time_max_abs_dV_dt=np.array(DT_Trans_table['Time_s'])[min_dV_dt_index]
        DT_trans_line=pd.Series([time_max_abs_dV_dt,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[min_dV_dt_index],'DT_trans_min_dV_dt'],
                                index=["Time_s","Membrane_potential_mV","Feature"])
        
    elif stim_amplitude > stimulus_baseline:

        max_dV_dt_index=np.nanargmax(potential_derivative_trace)
        time_max_abs_dV_dt=np.array(DT_Trans_table['Time_s'])[max_dV_dt_index]
        DT_trans_line=pd.Series([time_max_abs_dV_dt,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[max_dV_dt_index],'DT_trans_max_dV_dt'],
                                index=["Time_s","Membrane_potential_mV","Feature"])
    
    point_table=point_table.append(DT_trans_line,ignore_index=True)
    
    
    
    ##get the time of the last positive(negative) zero crossing dV/dt for a positive (negative) Istep, respectively
    if stim_amplitude <= stimulus_baseline:

        for reversed_index,elt in enumerate(potential_derivative_trace[::-1][1:]):
            previous_elt=potential_derivative_trace[::-1][reversed_index]
            if previous_elt >=0 and elt<0:
                last_zero_crossing_index=len(potential_derivative_trace)-1-reversed_index
                break
            else:
                last_zero_crossing_index=0
            
    elif stim_amplitude > stimulus_baseline:

        for reversed_index,elt in enumerate(potential_derivative_trace[::-1][1:]):
            previous_elt=potential_derivative_trace[::-1][reversed_index]
            if previous_elt <=0 and elt>0:
                last_zero_crossing_index=len(potential_derivative_trace)-1-reversed_index
                break
            else:
                last_zero_crossing_index=0
    
    time_zero_crossing=np.array(DT_Trans_table['Time_s'])[last_zero_crossing_index]
    
    DT_trans_line=pd.Series([time_zero_crossing,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[last_zero_crossing_index],'DT_trans_zero_crossing_dV_dt'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table=point_table.append(DT_trans_line,ignore_index=True)
    
    
    
    DT_trans=max(.002,abs(T_trans-time_max_abs_dV_dt),abs(T_trans-time_zero_crossing))

    
    #Get fit table : [T_seg_start; T_trans+T_seg_duration ]
    
    T_seg_start = T_trans+max(.002,2.5*DT_trans)
    T_seg_duration = max(.005,2.5*DT_trans)
    

    # Get a table (fit_table) of Time-Potential values between T_seg_start and T_seg_start+T_seg_duration
    fit_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
    fit_table=fit_table[fit_table['Time_s']>=(T_seg_start)]
    test_spike_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
    test_spike_table=stimulus_start_table[stimulus_start_table["Time_s"]>=T_trans]
    
    
    potential_spike_table=identify_spike_shorten(np.array(test_spike_table.Membrane_potential_mV),
                                   np.array(test_spike_table.Time_s),
                                   np.array(test_spike_table.Input_current_pA),
                                   np.array(test_spike_table.Time_s)[0],
                                   np.array(test_spike_table.Time_s)[-1],do_plot=False)
    
    
    
    if potential_spike_table == True:
    
         Bridge_Error_stim_start=np.nan
         
    else:
        
        
        #Use fit_table to fit either a 2nd order poly or Exponential to the membrane potential trace
        a,b,c,RMSE_poly=fitlib.fit_second_order_poly(fit_table,do_plot=False)
        A,tau,RMSE_expo=fitlib.fit_exponential(fit_table,do_plot=False)
        
        # Get a table between [T_trans and T_trans+T_seg_duration ] to estimate backward to T_Trans from the fit
        Post_trans_Init_Value_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
        Post_trans_Init_Value_table=Post_trans_Init_Value_table[Post_trans_Init_Value_table['Time_s']>=T_trans]
        
        #Use the best fit to extrapolate the membrane potential fit backward to T_Trans
        if RMSE_poly<=RMSE_expo:
            backward_prediction=a*((Post_trans_Init_Value_table.loc[:,'Time_s'])**2)+b*Post_trans_Init_Value_table.loc[:,'Time_s']+c
            Post_trans_init_value=a*((T_trans)**2)+b*T_trans+c
            
            backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
            backward_table['Legend']='Post_trans_init_value'
            
        elif RMSE_poly>RMSE_expo:
            backward_prediction=A*(np.exp(-(Post_trans_Init_Value_table.loc[:,'Time_s'])/tau))
            Post_trans_init_value=A*(np.exp(-(T_trans)/tau))
            backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
            backward_table['Legend']='Post_trans_init_value' 
        
        
        post_trans_init_value_line=pd.Series([T_trans,Post_trans_init_value,'Post_trans_init_value'],index=["Time_s","Membrane_potential_mV","Feature"])
        point_table=point_table.append(post_trans_init_value_line,ignore_index=True)
       

        #PreTrans_Est_table_to_fit is a table of Time_potential values between 1ms and 10ms before T_Trans,
        PreTrans_Est_table_to_fit=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_trans-.001)]
        PreTrans_Est_table_to_fit=PreTrans_Est_table_to_fit[PreTrans_Est_table_to_fit['Time_s']>=(T_trans-.01)]
        
        # #fit the potential values to a 2nd order polynomial
        a_bis,b_bis,c_bis,RMSE_poly_bis=fitlib.fit_second_order_poly(PreTrans_Est_table_to_fit,do_plot=False)
        
        # # Use the PreTrans_Est_table_to_fit fit to estimate the potential value at T_Trans = Pre_Trans_Final
        Pre_Trans_Final=a_bis*((T_trans)**2)+b_bis*T_trans+c_bis
        
        
        PreTrans_Est_table_to_fit['Legend']='Pre_Trans_fit'
      
        
        #Get the same table as PreTrans_Est_table_to_fit, but up to T_Trans, for plotting purpose
        PreTrans_Est_table_to_T_trans_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_trans)]
        PreTrans_Est_table_to_T_trans_table=PreTrans_Est_table_to_T_trans_table[PreTrans_Est_table_to_T_trans_table['Time_s']>=(T_trans-.01)]
        Pre_trans_Est_pred=a_bis*((PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"])**2)+b_bis*PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"]+c_bis
        #PreTrans_Est_table=PreTrans_Est_table_to_fit.loc[:,["Time_s","Membrane_potential_mV"]]
        PreTrans_Est_table=pd.DataFrame(np.column_stack((PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"],Pre_trans_Est_pred)),columns=["Time_s","Membrane_potential_mV"])
        PreTrans_Est_table['Legend']='PreTrans_Value'
        
        Pre_trans_final_value_line=pd.Series([T_trans,Pre_Trans_Final,'PreTrans_Value'],index=["Time_s","Membrane_potential_mV","Feature"])
        point_table=point_table.append(Pre_trans_final_value_line,ignore_index=True)
        
    
        Bridge_Error_stim_start = (Post_trans_init_value-Pre_Trans_Final)/I_step
        
        at_least_one_Bridge_Error=True
    ########################################################################################
    
    #Stimulus_end_window
    
    
    point_table_end=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Feature"])
    #Consider a 40msec windows centered on the specified stimulus start so +/- 20 msec and compute spline fit for current and potential
    stimulus_end_table=TPC_table[TPC_table["Time_s"]<(stim_end_time+.020)]
    stimulus_end_table=stimulus_end_table[stimulus_end_table['Time_s']>(stim_end_time-.020)]
    time_end_array=np.array(stimulus_end_table.loc[:,'Time_s'])
    
    
    current_end_trace_spline=scipy.interpolate.UnivariateSpline(stimulus_end_table.loc[:,'Time_s'],np.array(stimulus_end_table.loc[:,'Input_current_pA']))
    current_end_trace_spline.set_smoothing_factor(.999)
    current_end_trace_filtered=current_end_trace_spline(time_end_array)
    current_end_trace_filtered_derivative=calculate_dvdt ( current_end_trace_filtered , time_end_array )
    current_end_trace_filtered_derivative=np.insert(current_end_trace_filtered_derivative,0,current_end_trace_filtered_derivative[0])
    
    
    stimulus_end_table['Input_current_derivative_pA/s']=current_end_trace_filtered_derivative
    
    
    #determine T_trans_end
    T_trans_table_end=stimulus_end_table[stimulus_end_table["Time_s"]<(stim_end_time+.002)]
    T_trans_table_end=T_trans_table_end[T_trans_table_end['Time_s']>(stim_end_time-.002)]
   
    T_Trans_end_current_derivative=np.array(T_trans_table_end['Input_current_derivative_pA/s'])
    
    if stim_amplitude <= stimulus_baseline: # at the end of the stimulus
        
        max_abs_dI_dt_index_end = np.nanargmax(T_Trans_end_current_derivative)
        
    elif stim_amplitude>stimulus_baseline:
        max_abs_dI_dt_index_end = np.nanargmin(T_Trans_end_current_derivative)
        
   
    T_trans_end=np.array(T_trans_table_end.loc[:,'Time_s'])[max_abs_dI_dt_index_end]
   
   
   
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans-0.006 and T_trans-0.002 (and vice-versa)
    Pre_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end-.002]
    Pre_Trans_Median_table_end=Pre_Trans_Median_table_end[Pre_Trans_Median_table_end['Time_s']>=T_trans_end-.006]
    Pre_Trans_Median_end=np.median(Pre_Trans_Median_table_end['Input_current_pA'])
    
    Post_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']>=T_trans_end+.002]
    Post_Trans_Median_table_end=Post_Trans_Median_table_end[Post_Trans_Median_table_end['Time_s']<=T_trans_end+.006]
    Post_Trans_Median_end=np.median(Post_Trans_Median_table_end['Input_current_pA'])
    
    I_step_end=Post_Trans_Median_end-Pre_Trans_Median_end
   
    
   
   
    # T_trans_line_end=pd.Series([T_trans_end,np.array(T_trans_table_end.loc[:,'Membrane_potential_mV'])[max_abs_dI_dt_index_end],'T_trans_end'],index=["Time_s","Membrane_potential_mV","Feature"])
    # point_table_end=point_table_end.append(T_trans_line_end,ignore_index=True)
    
    
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans_end-0.006 and T_trans_end-0.002 (and vice-versa)
    Pre_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end-.002]
    Pre_Trans_Median_table_end=Pre_Trans_Median_table_end[Pre_Trans_Median_table_end['Time_s']>=T_trans_end-.006]
    Pre_Trans_Median_end=np.median(Pre_Trans_Median_table_end['Input_current_pA'])
    
    Post_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']>=T_trans_end+.002]
    Post_Trans_Median_table_end=Post_Trans_Median_table_end[Post_Trans_Median_table_end['Time_s']<=T_trans_end+.006]
    Post_Trans_Median_end=np.median(Post_Trans_Median_table_end['Input_current_pA'])
    
    I_step_end=Post_Trans_Median_end-Pre_Trans_Median_end
    
    #determine DT_trans, the time scale of response transient to avoid contaminated period;
   
    ## get time of the max (min) dV/dt, for a positive (negative) Istep, respectively
    DT_Trans_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end].reset_index(drop=True).copy()
   
   
    DT_Trans_table_end=DT_Trans_table_end[DT_Trans_table_end['Time_s']>=(T_trans_end-.003)]
    potential_derivative_trace=np.array(DT_Trans_table_end.loc[:,'Potential_first_time_derivative_mV/s'])


    
    if stim_amplitude <= stimulus_baseline:
   
        max_dV_dt_index=np.nanargmax(potential_derivative_trace)
        time_max_abs_dV_dt_end=np.array(DT_Trans_table_end['Time_s'])[max_dV_dt_index]
        DT_trans_line_end=pd.Series([time_max_abs_dV_dt_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[max_dV_dt_index],'DT_trans_min_dV_dt'],
                                index=["Time_s","Membrane_potential_mV","Feature"])
        
    elif stim_amplitude > stimulus_baseline:
   
        min_dV_dt_index=np.nanargmin(potential_derivative_trace)
        time_max_abs_dV_dt_end=np.array(DT_Trans_table_end['Time_s'])[min_dV_dt_index]
        DT_trans_line_end=pd.Series([time_max_abs_dV_dt_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[min_dV_dt_index],'DT_trans_max_dV_dt'],
                                index=["Time_s","Membrane_potential_mV","Feature"])
    
    point_table_end=point_table_end.append(DT_trans_line_end,ignore_index=True)
    
    
    
    ##get the time of the last positive(negative) zero crossing dV/dt for a positive (negative) Istep, respectively
    if stim_amplitude <= stimulus_baseline:
   
        for index,elt in enumerate(potential_derivative_trace[1:]):
            previous_elt=potential_derivative_trace[index]
            if previous_elt <=0 and elt>0:
                last_zero_crossing_index=index
                break
            else:
                last_zero_crossing_index=len(potential_derivative_trace)-1
            
    elif stim_amplitude > stimulus_baseline:
   
        for index,elt in enumerate(potential_derivative_trace[1:]):
            previous_elt=potential_derivative_trace[index]
            if previous_elt >=0 and elt<0:
                last_zero_crossing_index=index
                break
            else:
                last_zero_crossing_index=len(potential_derivative_trace)-1
    
    time_zero_crossing_end=np.array(DT_Trans_table_end['Time_s'])[last_zero_crossing_index]
    
    DT_trans_line_end=pd.Series([time_zero_crossing_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[last_zero_crossing_index],'DT_trans_zero_crossing_dV_dt'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table_end=point_table_end.append(DT_trans_line_end,ignore_index=True)
    
    DT_trans_end=max(.002,abs(T_trans_end-time_max_abs_dV_dt_end),abs(T_trans_end-time_zero_crossing_end))
    
    #Get fit table : [T_trans_end-T_seg_duration_end ; T_seg_start_end]
    
    T_seg_start_end = T_trans_end-max(.002,2.5*DT_trans_end)
    T_seg_duration_end = max(.005,2.5*DT_trans_end)
    
   
    # Get a table (fit_table_end) of Time-Potential values between T_seg_start_end and T_seg_start_end-T_seg_duration_end
    fit_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
    
    fit_table_end=fit_table_end[fit_table_end['Time_s']<=(T_seg_start_end)]
    
    test_spike_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
    test_spike_table_end=stimulus_end_table[stimulus_end_table["Time_s"]<=T_trans_end]
    
    potential_spike_table=identify_spike_shorten(np.array(test_spike_table_end.Membrane_potential_mV),
                                    np.array(test_spike_table_end.Time_s),
                                    np.array(test_spike_table_end.Input_current_pA),
                                    np.array(test_spike_table_end.Time_s)[0],
                                    np.array(test_spike_table_end.Time_s)[-1],do_plot=False)
    if potential_spike_table == True:
    #if np.nanmax(fit_table_end['Potential_first_time_derivative_mV/s']) >20. or np.nanmax(fit_table_end['Potential_first_time_derivative_mV/s']) <-20.:
        Bridge_Error_stim_end=np.nan


    else:
        
        #Use fit_table_end to fit either a 2nd order poly or Exponential to the membrane potential trace
        a,b,c,RMSE_poly=fitlib.fit_second_order_poly(fit_table_end,do_plot=False)
        A,tau,RMSE_expo=fitlib.fit_exponential(fit_table_end,do_plot=False)
        
        # Get a table between [T_trans_end and T_trans_end+T_seg_duration_end ] to estimate backward to T_trans_end from the fit
        Pre_trans_Init_Value_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
        Pre_trans_Init_Value_table_end=Pre_trans_Init_Value_table_end[Pre_trans_Init_Value_table_end['Time_s']<=T_trans_end]
        
        #Use the best fit to extrapolate the membrane potential fit backward to T_trans_end
        if RMSE_poly<=RMSE_expo:
            backward_prediction_end=a*((Pre_trans_Init_Value_table_end.loc[:,'Time_s'])**2)+b*Pre_trans_Init_Value_table_end.loc[:,'Time_s']+c
            Pre_trans_init_value_end=a*((T_trans_end)**2)+b*T_trans_end+c
            
            backward_table_end=pd.DataFrame(np.column_stack((Pre_trans_Init_Value_table_end.loc[:,'Time_s'],backward_prediction_end)),columns=["Time_s","Membrane_potential_mV"])
            backward_table_end['Legend']='Pre_trans_init_value'
            
        elif RMSE_poly>RMSE_expo:
            backward_prediction_end=A*(np.exp(-(Pre_trans_Init_Value_table_end.loc[:,'Time_s'])/tau))
            Pre_trans_init_value_end=A*(np.exp(-(T_trans_end)/tau))
            backward_table_end=pd.DataFrame(np.column_stack((Pre_trans_Init_Value_table_end.loc[:,'Time_s'],backward_prediction_end)),columns=["Time_s","Membrane_potential_mV"])
            backward_table_end['Legend']='Pre_trans_init_value' 
        
    
        pre_trans_init_value_line=pd.Series([T_trans_end,Pre_trans_init_value_end,'Pre_trans_init_value'],index=["Time_s","Membrane_potential_mV","Feature"])
        point_table_end=point_table_end.append(pre_trans_init_value_line,ignore_index=True)
       
        #Corrected Version Post trans.Final
        #PostTrans_Est_table_to_fit is a table of Time_potential values between 1ms and 10ms before T_trans_end,
        PostTrans_Est_table_to_fit=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_trans_end+.001)]
        PostTrans_Est_table_to_fit=PostTrans_Est_table_to_fit[PostTrans_Est_table_to_fit['Time_s']<(T_trans_end+.01)]
        
        #fit the potential values to a 2nd order polynomial
        a_bis,b_bis,c_bis,RMSE_poly_bis=fitlib.fit_second_order_poly(PostTrans_Est_table_to_fit,do_plot=False)
        
        # Use the PostTrans_Est_table_to_fit fit to estimate the potential value at T_trans_end = Post_Trans_Final
        Post_Trans_Final=a_bis*((T_trans_end)**2)+b_bis*T_trans_end+c_bis
        PostTrans_Est_table_to_fit['Legend']='PostTrans_fit'
        #Get the same table as PostTrans_Est_table_to_fit, but up to T_trans_end, for plotting purpose
        PostTrans_Est_table_to_T_trans_table=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_trans_end)]
        PostTrans_Est_table_to_T_trans_table=PostTrans_Est_table_to_T_trans_table[PostTrans_Est_table_to_T_trans_table['Time_s']<(T_trans_end+.01)]
        PostTrans_Est_pred=a_bis*((PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"])**2)+b_bis*PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"]+c_bis
        
        PostTrans_Est_table=pd.DataFrame(np.column_stack((PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"],PostTrans_Est_pred)),columns=["Time_s","Membrane_potential_mV"])
        PostTrans_Est_table['Legend']='PostTrans_Est'
        
        Post_trans_final_value_line=pd.Series([T_trans_end,Post_Trans_Final,'PostTrans_Est'],index=["Time_s","Membrane_potential_mV","Feature"])
        point_table_end=point_table_end.append(Post_trans_final_value_line,ignore_index=True)
        
        
        #fit_table_end=pd.concat([backward_table, PreTrans_Est_table], axis=0)
        
        #Bridge_Error= (Post_trans_init_value-Pre_Trans_Final_Value)/(stim_amplitude-stimulus_baseline)
        
       
         
     ########################################################################################
       # Pre_Trans_Final_Value=a_bis*((PreTrans_Est_table_to_fit.iloc[-1,0])**2)+b_bis*PreTrans_Est_table_to_fit.iloc[-1,0]+c_bis
        
        Bridge_Error_stim_end = (Post_Trans_Final-Pre_trans_init_value_end)/I_step_end
        at_least_one_Bridge_Error=True
    
    if at_least_one_Bridge_Error==True:
        
        Bridge_Error=np.nanmin(np.array([Bridge_Error_stim_start,Bridge_Error_stim_end]))
    else: 
        Bridge_Error = np.nan

    ########################################################################################
    exponential_fit_table=TPC_table[TPC_table["Time_s"]<=(T_trans+.152)]
    exponential_fit_table=exponential_fit_table[exponential_fit_table['Time_s']>=(T_trans+.002)]
    
    ###check for spike in this time period
    
    potential_spike_expo=identify_spike_shorten(np.array(exponential_fit_table.Membrane_potential_mV),
                                    np.array(exponential_fit_table.Time_s),
                                    np.array(exponential_fit_table.Input_current_pA),
                                    np.array(exponential_fit_table.Time_s)[0],
                                    np.array(exponential_fit_table.Time_s)[-1],do_plot=False)
    
    if potential_spike_expo==False:
        best_A,best_tau,membrane_SS=fitlib.fit_membrane_trace(exponential_fit_table,np.array(exponential_fit_table.Time_s)[0],np.array(exponential_fit_table.Time_s)[-1],do_plot=do_plot)

        pred=fitlib.time_cst_model(np.array(exponential_fit_table.Time_s),best_A,best_tau,membrane_SS)
        squared_error = np.square((np.array(exponential_fit_table.Membrane_potential_mV) - pred))
        sum_squared_error = np.sum(squared_error)
        current_RMSE = np.sqrt(sum_squared_error / len(np.array(exponential_fit_table.Membrane_potential_mV)))

        if current_RMSE/abs(Pre_Trans_Final-membrane_SS) >= -.003  and current_RMSE/abs(Pre_Trans_Final-membrane_SS) <= .003:
            
            response_SS=membrane_SS
        else:
            SS_table=TPC_table[TPC_table["Time_s"]<=(T_trans+.160)]
            SS_table=SS_table[SS_table['Time_s']>=(T_trans+.050)]
            response_SS=np.median(np.array(SS_table['Membrane_potential_mV']))
            
        membrane_time_cst=best_tau*1e3 #convert to ms
        
        if not np.isnan(Bridge_Error):
            R_in=((response_SS-Pre_Trans_Final)/I_step)-Bridge_Error
            
            
            V_rest_table=TPC_table[TPC_table["Time_s"]<=(T_trans)]
            V_rest_table=V_rest_table[V_rest_table["Time_s"]>=(T_trans-.100)]
            
            V_rest=np.mean(np.array(V_rest_table["Membrane_potential_mV"]))-np.mean(np.array(V_rest_table["Input_current_pA"]))*R_in
            R_in*=1e3#convert to MOhms
        else:
            
            R_in=np.nan
            V_rest=np.nan
            
    else:
        membrane_time_cst=np.nan
        SS_table=TPC_table[TPC_table["Time_s"]<=(T_trans+.160)]
        SS_table=SS_table[SS_table['Time_s']>=(T_trans+.050)]
        response_SS=np.median(np.array(SS_table['Membrane_potential_mV']))
        if not np.isnan(Bridge_Error):
            R_in=((response_SS-Pre_Trans_Final)/I_step)-Bridge_Error
            
            
            V_rest_table=TPC_table[TPC_table["Time_s"]<=(T_trans)]
            V_rest_table=V_rest_table[V_rest_table["Time_s"]>=(T_trans-.100)]
            
            V_rest=np.mean(np.array(V_rest_table["Membrane_potential_mV"]))-np.mean(np.array(V_rest_table["Input_current_pA"]))*R_in
            R_in*=1e3#convert to MOhms
        else:
            
            R_in=np.nan
            V_rest=np.nan
            
    
    
    
    
    
    if do_plot:
        my_plot=ggplot(TPC_table,aes(x='Time_s',y='Membrane_potential_mV'))+geom_line(color='black')
        my_plot+= geom_line(fit_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red',size=.7)
        my_plot_end=ggplot(TPC_table,aes(x='Time_s',y='Membrane_potential_mV'))+geom_line(color='black')
        my_plot_end+= geom_line(fit_table_end,aes(x='Time_s',y='Membrane_potential_mV'),color='red',size=.7)
        if np.isnan(Bridge_Error_stim_start) == False :
            
            my_plot+= geom_line(backward_table,aes(x='Time_s',y='Membrane_potential_mV',color="Legend"))
            my_plot+=geom_line(PreTrans_Est_table_to_fit,aes(x='Time_s',y='Membrane_potential_mV'),color='yellow',size=.7)
            my_plot+=geom_line(PreTrans_Est_table,aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot+=geom_point(point_table,aes(x='Time_s',y='Membrane_potential_mV',color="Feature"))
            my_plot+=xlim(min(np.array(PreTrans_Est_table['Time_s']))-.005,max(np.array(fit_table['Time_s']))+.005)
        else:
            my_plot+= geom_line(test_spike_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red',linetype='dashed',size=.9)
            
            
        if np.isnan(Bridge_Error_stim_end) == False:
            
            
            my_plot_end+=geom_line(backward_table_end,aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot_end+=geom_line(PostTrans_Est_table_to_fit,aes(x='Time_s',y='Membrane_potential_mV'),color='yellow',size=.7)
            my_plot_end+=geom_line(PostTrans_Est_table,aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot_end+=geom_point(point_table_end,aes(x='Time_s',y='Membrane_potential_mV',color="Feature"))
            my_plot_end+=xlim(min(np.array(fit_table_end['Time_s']))-.005,max(np.array(PostTrans_Est_table['Time_s']))+.005)
            
        else: 
            my_plot_end+= geom_line(test_spike_table_end,aes(x='Time_s',y='Membrane_potential_mV'),color='red',linetype='dashed',size=.9)
            #my_plot+=xlim(stim_start_time-.03,stim_start_time+.03)
        my_plot+=geom_vline(xintercept=T_trans,color='purple')
        my_plot_end+=geom_vline(xintercept=T_trans_end,color='purple')
        my_plot_end+=xlim(stim_end_time-.03,stim_end_time+.03)
        my_plot+=ggtitle(str("Membrane_potential_trace_stim_start\n\nBridge_Error="+str(round(Bridge_Error_stim_start,3))+" GOhms"))
        my_plot_end+=ggtitle(str("Membrane_potential_trace_stim_end\n\nBridge_Error="+str(round(Bridge_Error_stim_end,3))+" GOhms"))
        
            
            
       
        print(my_plot)
        print(my_plot_end)
        
        stimulus_start_table['Filtered_current_trace']=current_trace_filtered

        current_plot=ggplot(TPC_table,aes(x='Time_s',y='Input_current_pA'))+geom_line(color='black')
        current_plot+=geom_line(stimulus_start_table,aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot+=geom_line(stimulus_start_table,aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
        current_plot+=xlim(stim_start_time-.03,stim_start_time+.03)
        current_plot+=ggtitle(str("Input_Current_trace_Stim_start\n\nBridge_Error="+str(round(Bridge_Error_stim_start,3))+" GOhms"))

        current_plot+=geom_hline(yintercept=Pre_Trans_Median,color='blue')
        current_plot+=geom_hline(yintercept=Post_Trans_Median,color='red')
        current_plot+=geom_vline(xintercept=T_trans,color='purple')
        print(current_plot)
        
        
        
        
        stimulus_start_table['Filtered_current_trace']=current_trace_filtered
        current_plot=ggplot(TPC_table,aes(x='Time_s',y='Input_current_pA'))+geom_line(color='black')
        current_plot+=geom_line(stimulus_end_table,aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot+=geom_line(stimulus_end_table,aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
       
        current_plot+=ggtitle(str("Input_Current_trace_Stim_end\n\nBridge_Error="+str(round(Bridge_Error_stim_end,3))+" GOhms"))
        
        stimulus_end_table['Filtered_current_trace']=current_end_trace_filtered
        current_plot+=geom_hline(yintercept=Post_Trans_Median_end,color='blue')
        current_plot+=geom_hline(yintercept=Pre_Trans_Median_end,color='red')
        current_plot+=geom_line(stimulus_end_table,aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot+=geom_line(stimulus_end_table,aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
        current_plot+=geom_vline(xintercept=T_trans_end,color='purple')
        current_plot+=xlim(stim_end_time-.03,stim_end_time+.03)
        print(current_plot)
        
    return Bridge_Error,membrane_time_cst,R_in,V_rest
    
    
        
        
    
    #print(dV_dt_trans,max_abs_dI_dt_index,T_trans,time_max_abs_dV_dt,time_zero_crossing,DT_trans)
    
    
def old_estimate_bridge_error(TPC_table,stim_amplitude,stim_start_time,stim_end_time):
    
    
    TPC_table=TPC_table.reset_index(drop=True)
    point_table=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Feature"])
    current_derivative=calculate_dvdt(TPC_table.loc[:,'Input_current_pA'],TPC_table.loc[:,'Time_s'])
    current_derivative=np.insert(current_derivative,0,current_derivative[0])
    TPC_table["Input_current_derivative_pA/s"]=current_derivative
    
    
    stimulus_start_table=TPC_table[TPC_table["Time_s"]<(stim_start_time+.005)]
    stimulus_start_table=stimulus_start_table[stimulus_start_table['Time_s']>(stim_start_time-.005)]
    
    potential_trace=np.array(stimulus_start_table.loc[:,'Membrane_potential_mV'])
    time_array=np.array(stimulus_start_table.loc[:,'Time_s'])
    start_time_index = np.argmin(abs(time_array - stim_start_time))
    end_time_index = np.argmin(abs(time_array - stim_end_time))
    stimulus_baseline=np.mean(TPC_table.loc[:(start_time_index-1),'Input_current_pA'])
    current_derivative=np.array(stimulus_start_table.loc[:,'Input_current_derivative_pA/s'])
    #print("baseline=",stimulus_baseline)
    #determine T_trans
    if stim_amplitude <= stimulus_baseline:

        min_dI_dt = np.nanmin((np.array(stimulus_start_table.loc[:,'Input_current_derivative_pA/s'])))
        max_abs_dI_dt_index = np.argmin(abs(current_derivative - min_dI_dt))
    elif stim_amplitude>stimulus_baseline:

        max_dI_dt = np.nanmax((np.array(stimulus_start_table.loc[:,'Input_current_derivative_pA/s'])))
        max_abs_dI_dt_index = np.argmin(abs(current_derivative - max_dI_dt))
        print("max_abs_dI_dt_index=",max_abs_dI_dt_index)
        
    T_trans=time_array[max_abs_dI_dt_index]
    T_trans_line=pd.Series([T_trans,potential_trace[max_abs_dI_dt_index],'T_trans'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table=point_table.append(T_trans_line,ignore_index=True)
    #determine DT_trans
    print('T_trans=',T_trans)
    ## get time of the max (min) dV/dt, for a positive (negative) Istep, respectively
    potential_derivative_trace=np.array(stimulus_start_table.loc[:,'Potential_first_time_derivative_mV/s'])
    if stim_amplitude <= stimulus_baseline:
        min_dV_dt=np.nanmin(((np.array(stimulus_start_table.loc[:,'Potential_first_time_derivative_mV/s']))))
        min_dV_dt_index=np.argmin(abs(potential_derivative_trace - min_dV_dt))
        time_max_abs_dV_dt=time_array[min_dV_dt_index]
        DT_trans_line=pd.Series([time_max_abs_dV_dt,potential_trace[min_dV_dt_index],'DT_trans_min_dV_dt'],index=["Time_s","Membrane_potential_mV","Feature"])
        
    elif stim_amplitude > stimulus_baseline:
        
        max_dV_dt=np.nanmax(((np.array(stimulus_start_table.loc[:,'Potential_first_time_derivative_mV/s']))))
        max_dV_dt_index=np.argmin(abs(potential_derivative_trace - max_dV_dt))
        print("max_dV_dt_index=",max_dV_dt_index)
        
        time_max_abs_dV_dt=time_array[max_dV_dt_index]
        print("time_max_abs_dV_dt=",time_max_abs_dV_dt)
        DT_trans_line=pd.Series([time_max_abs_dV_dt,potential_trace[max_dV_dt_index],'DT_trans_max_dV_dt'],index=["Time_s","Membrane_potential_mV","Feature"])
    
    point_table=point_table.append(DT_trans_line,ignore_index=True)
    
    sub_potential_derivative_trace=potential_derivative_trace[:max_abs_dI_dt_index]
    
    
    if stim_amplitude <= stimulus_baseline:

        for reversed_index,elt in enumerate(sub_potential_derivative_trace[::-1][1:]):
            previous_elt=sub_potential_derivative_trace[::-1][reversed_index]
            if previous_elt >=0 and elt<0:
                last_zero_crossing_index=len(sub_potential_derivative_trace)-1-reversed_index
                break
            
            
    elif stim_amplitude > stimulus_baseline:
        
        for reversed_index,elt in enumerate(sub_potential_derivative_trace[::-1][1:]):
            previous_elt=sub_potential_derivative_trace[::-1][reversed_index]
            if previous_elt <=0 and elt>0:
                last_zero_crossing_index=len(sub_potential_derivative_trace)-1-reversed_index
                break
         
    time_zero_crossing=time_array[last_zero_crossing_index]
    DT_trans_line=pd.Series([time_zero_crossing,potential_trace[last_zero_crossing_index],'DT_trans_zero_crossing_dV_dt'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table=point_table.append(DT_trans_line,ignore_index=True)
    DT_trans=np.mean((abs(T_trans-time_max_abs_dV_dt),abs(T_trans-time_zero_crossing)))
    print('time_max_abs_dV_dt=',time_max_abs_dV_dt)
    print('time_zero_crossing=',time_zero_crossing)
    print('last_zero_crossing_index=',last_zero_crossing_index)
    #Get fit table : [T_trans+2.5*DT_trans; T_trans+2.5*DT_trans+ 5*DT_trans ]

    fit_table=TPC_table[TPC_table["Time_s"]<(T_trans+2.5*DT_trans+ 5*DT_trans)]
    fit_table=fit_table[fit_table['Time_s']>(T_trans+2.5*DT_trans)]
    print("T_trans+2.5*DT_trans=",T_trans+2.5*DT_trans)
    print("T_trans+2.5*DT_trans+ 5*DT_trans=",T_trans+2.5*DT_trans+ 5*DT_trans)
    
    
    print(fit_table.iloc[:,:3])
    a,b,c,RMSE_poly=fitlib.fit_second_order_poly(fit_table,do_plot=False)
    A,tau,RMSE_expo=fitlib.fit_exponential(fit_table,do_plot=False)
    
    Post_trans_Init_Value_table=TPC_table[TPC_table["Time_s"]<(T_trans+2.5*DT_trans+ 5*DT_trans)]
    Post_trans_Init_Value_table=Post_trans_Init_Value_table[Post_trans_Init_Value_table['Time_s']>=T_trans]

    if RMSE_poly<=RMSE_expo:
        backward_prediction=a*((Post_trans_Init_Value_table.loc[:,'Time_s'])**2)+b*Post_trans_Init_Value_table.loc[:,'Time_s']+c
        Post_trans_init_value=a*((T_trans)**2)+b*T_trans+c
        
        backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
        backward_table['Legend']='Fit_curve'
        
    elif RMSE_poly>RMSE_expo:
        backward_prediction=A*(np.exp(-(Post_trans_Init_Value_table.loc[:,'Time_s'])/tau))
        Post_trans_init_value=A*(np.exp(-(T_trans)/tau))
        backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
        backward_table['Legend']='Fit_curve' 
        
    post_trans_init_value_line=pd.Series([T_trans,Post_trans_init_value,'Post_trans_init_value'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table=point_table.append(post_trans_init_value_line,ignore_index=True)

    
    PreTrans_Est_table=TPC_table[TPC_table["Time_s"]<=(T_trans-.001)]
    PreTrans_Est_table=PreTrans_Est_table[PreTrans_Est_table['Time_s']>(T_trans-.01)]
    
    
    a_bis,b_bis,c_bis,RMSE_poly_bis=fitlib.fit_second_order_poly(PreTrans_Est_table,do_plot=False)

    
    Pre_Trans_Final_Value=a_bis*((PreTrans_Est_table.iloc[-1,0])**2)+b_bis*PreTrans_Est_table.iloc[-1,0]+c_bis
    Pre_trans_Est_pred=a_bis*((PreTrans_Est_table.loc[:,"Time_s"])**2)+b_bis*PreTrans_Est_table.loc[:,"Time_s"]+c_bis
    
    PreTrans_Est_table=pd.DataFrame(np.column_stack((PreTrans_Est_table.loc[:,"Time_s"],Pre_trans_Est_pred)),columns=["Time_s","Membrane_potential_mV"])
    PreTrans_Est_table['Legend']='PreTrans_Est'
    
    Pre_trans_final_value_line=pd.Series([PreTrans_Est_table.iloc[-1,0],Pre_Trans_Final_Value,'Pre_Trans_Final_Value'],index=["Time_s","Membrane_potential_mV","Feature"])
    point_table=point_table.append(Pre_trans_final_value_line,ignore_index=True)
    Fit_table=pd.concat([backward_table, PreTrans_Est_table], axis=0)
    
    Bridge_Error= (Post_trans_init_value-Pre_Trans_Final_Value)/(stim_amplitude-stimulus_baseline)
    print("T_trans=",T_trans)
    
    my_plot=ggplot(TPC_table,aes(x='Time_s',y='Membrane_potential_mV'))+geom_line(color='black')
    my_plot+=geom_line(Fit_table,aes(x='Time_s',y='Membrane_potential_mV',color='Legend')) 
    my_plot+= geom_line(backward_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red')
    my_plot+= geom_line(Post_trans_Init_Value_table,aes(x='Time_s',y='Membrane_potential_mV'),color='blue')
    my_plot+=geom_point(point_table,aes(x='Time_s',y='Membrane_potential_mV',color="Feature"))
    #my_plot+=ylim(-70.5,-60) 
    my_plot+=xlim(1.99,2.01)
    my_plot+=geom_vline(xintercept=T_trans)
    my_plot+=ggtitle("Previous_membrane_trace")
    print(my_plot)
    print("old_I_Step=",stim_amplitude-stimulus_baseline)
    second_plot=ggplot(TPC_table,aes(x='Time_s',y='Input_current_pA'))+geom_line(color='black')
    second_plot+=xlim(1.95,2.05)
    second_plot+=ggtitle("Previous_Input_current_trace")
    second_plot+=geom_hline(yintercept=stim_amplitude,color='red')
    second_plot+=geom_hline(yintercept=stimulus_baseline,color='blue')
    
    print(second_plot)
    return Bridge_Error
    # if abs(Bridge_Error) <0.001:
    #     print('Last_run, Rb=',Bridge_Error)
    #     return Bridge_Error
    # else:
    #     print("NEW_RUN,Bridge_error=",Bridge_Error)
        
    #     time_trace=np.array(TPC_table['Time_s'])
    #     start_time_index = np.argmin(abs(time_trace - stim_start_time))
    #     end_time_index = np.argmin(abs(time_trace - stim_end_time))
    #     corrected_potential_trace=np.array(TPC_table['Membrane_potential_mV'])
        
    #     corrected_potential_trace[start_time_index:end_time_index]=corrected_potential_trace[start_time_index:end_time_index]+Bridge_Error*(stimulus_baseline-stim_amplitude)
        
    #     new_TPC_table=create_TPC(time_trace=time_trace,
    #                           potential_trace=corrected_potential_trace,
    #                           current_trace=np.array(TPC_table['Input_current_pA']),
    #                           filter=2.)
        
    #     new_TPC_table=new_TPC_table.reset_index(drop=True)
    #     new_TPC_table.loc[0,"Potential_first_time_derivative_mV/s"]=new_TPC_table.loc[1,"Potential_first_time_derivative_mV/s"]
    #     return Bridge_Error+estimate_bridge_error(new_TPC_table,stim_amplitude,stim_start_time,stim_end_time)
    
    
        
