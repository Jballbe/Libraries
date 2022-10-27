#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:53:26 2022

@author: julienballbe
"""

import webbrowser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
import time
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle

import scipy
from scipy.stats import linregress
from scipy import optimize
import random

import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import warnings
import pandas
import os
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel,GaussianModel
from lmfit import Parameters, Minimizer,fit_report
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
from plotnine.labels import xlab
from plotnine.coords import coord_cartesian
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.interpolate import splrep, splev
from scipy.misc import derivative
from scipy import io

from allensdk.core.cell_types_cache import CellTypesCache

#%%
ctc= CellTypesCache(manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")

#%%
def fit_specimen_fi_slope(stim_amps, avg_rates):
    """
    Fit the rate and stimulus amplitude to a line and return the slope of the fit.

    Parameters
    ----------
    stim_amps: array of sweeps amplitude in mA
    avg_rates: array of sweeps avergae firing rate in Hz
    Returns
    -------
    m: f-I curve slope for the specimen
    c:f-I curve intercept for the specimen

    """

    x = stim_amps
    y = avg_rates

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c

def get_stim_freq_table(SF_table, response_time):
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    SF_table['Frequency_Hz']=0
    for current_sweep in np.array(SF_table.loc[:,"Sweep"]):
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        SF_table.loc[current_sweep,'Frequency_Hz']=(df[df['Time_s']<(SF_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])/response_time
        
    SF_table=SF_table.loc[:,['Sweep', 'Stim_amp_pA', 'Frequency_Hz']]
    return SF_table

def data_pruning (stim_freq_table):
    
    stim_freq_table=stim_freq_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
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
        do_fit=False
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

def normalized_root_mean_squared_error(true, pred,pred_extended):
    #Normalization by the interquartile range
    squared_error = np.square((true - pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / true.size)
    Q1=np.percentile(pred_extended,25)
    Q3=np.percentile(pred_extended,75)

    nrmse_loss = rmse/(Q3-Q1)
    return nrmse_loss

def hill_function(x, Amplitude,Hill_coef,Half_cst):
    return Amplitude*((x**(Hill_coef))/((Half_cst**Hill_coef)+(x**(Hill_coef))))


def fit_IO_curve(stimulus_frequency_table,do_plot=False):
    
    try:
         stimulus_frequency_table=stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
         x_shift=abs(min(stimulus_frequency_table['Stim_amp_pA']))
         stimulus_frequency_table.loc[:,'Stim_amp_pA']=stimulus_frequency_table.loc[:,'Stim_amp_pA']+x_shift
         stimulus_frequency_table=stimulus_frequency_table.reset_index(drop=True)
         #stimulus_frequency_table.index=[x for x in range(stimulus_frequency_table.shape[0]+1)]

         x_data=stimulus_frequency_table.loc[:,'Stim_amp_pA']
         y_data=stimulus_frequency_table.loc[:,"Frequency_Hz"]

         
         #get initial estimate of parameters for single sigmoid fit
         without_zero_index=np.flatnonzero(y_data)[0]
         median_firing_rate_index=np.argmax(y_data>np.median(y_data.iloc[without_zero_index:]))

         #Get the stimulus amplitude correspondingto the median non-zero firing rate
         x0=x_data.iloc[median_firing_rate_index]

         
         max_freq_step_index=np.argmax(y_data.diff())
         freq_step_array=y_data.diff()
         stim_step_array=x_data.diff()
         
         max_freq_step=freq_step_array[max_freq_step_index]
         max_stim_step=stim_step_array[max_freq_step_index]
         normalized_step=max_freq_step/max_stim_step
         
         sub_x_data=x_data.iloc[without_zero_index:]
         sub_y_data=y_data.iloc[without_zero_index:]

         if normalized_step>=1.5:
             obs='Max_step_too_high'
             best_Amplitude=np.nan
             best_Half_cst=np.nan
             best_Hill_coef=np.nan
             QNRMSE=np.nan
             x_shift=np.nan
             Gain=np.nan
             Threshold=np.nan
             Saturation=np.nan
             
         
         else: #If the highest step is not too high, try to fit 
             Gain=np.nan
             Threshold=np.nan
             Saturation=np.nan
             
             hillModel=Model(hill_function)
             
             hill_parameters=Parameters()

             hill_parameters.add("Half_cst",value=x0,min=0)
             hill_parameters.add('Hill_coef',value=2,min=0)
             hill_parameters.add('Amplitude',value=max(y_data))
             
            
             result = hillModel.fit(y_data, hill_parameters, x=x_data)

             best_Amplitude=result.best_values['Amplitude']
             best_Half_cst=result.best_values['Half_cst']
             best_Hill_coef=result.best_values['Hill_coef']
             
             new_x_data=pd.Series(np.arange(min(x_data),max(x_data),1))
             predicted_y_data=hill_function(new_x_data, best_Amplitude, best_Hill_coef, best_Half_cst)
             
             new_x_data_without_zero=pd.Series(np.arange(min(sub_x_data),max(sub_x_data),1))
             pred_without_zero=pd.Series(hill_function(new_x_data_without_zero, best_Amplitude, best_Hill_coef, best_Half_cst))
             
             pred=[]
             for elt in sub_x_data:
                 pred.append(hill_function(elt, best_Amplitude, best_Hill_coef, best_Half_cst))
                 
             pred=np.array(pred)
             QNRMSE=normalized_root_mean_squared_error(sub_y_data,pred,pred_without_zero)
             
             
             if QNRMSE >0.5:
                 obs='QNRMSE_too_high'
                 best_Amplitude=np.nan
                 best_Half_cst=np.nan
                 best_Hill_coef=np.nan
                 mylinetype='dashed'
             else:
                 obs='--'
                 mylinetype="solid"
                 
                 twentyfive_index=next(x for x, val in enumerate(predicted_y_data) if val >(0.25*max(predicted_y_data)))
                 seventyfive_index=next(x for x, val in enumerate(predicted_y_data) if val >(0.75*max(predicted_y_data)))
                 
                 Gain,Intercept_shift=fit_specimen_fi_slope(new_x_data.iloc[twentyfive_index:seventyfive_index],predicted_y_data.iloc[twentyfive_index:seventyfive_index])
                 Threshold_shift=(0-Intercept_shift)/Gain
                 Intercept=Gain*x_shift+Intercept_shift
                 Threshold=Threshold_shift-x_shift
                 
                 my_derivative=np.array(derivative(hill_function,new_x_data,dx=1e-1,args=(best_Amplitude,best_Hill_coef,best_Half_cst)))
                 end_slope=np.mean(my_derivative[-100:])
                 Saturation=np.nan
                 
                 if end_slope <=0.001:
                     Saturation=np.mean(predicted_y_data[-100:])

             if do_plot == True:
                 stimulus_frequency_table.loc[:,'Stim_amp_pA']=stimulus_frequency_table.loc[:,'Stim_amp_pA']-x_shift
                 new_x_data-=x_shift
                 model_table=pd.DataFrame(np.column_stack((new_x_data,predicted_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
                 my_plot=ggplot(stimulus_frequency_table,aes(x=stimulus_frequency_table["Stim_amp_pA"],y=stimulus_frequency_table["Frequency_Hz"]))+geom_point()
                 my_plot+=geom_line(model_table,aes(x=model_table["Stim_amp_pA"],y=model_table['Frequency_Hz']),color='red',linetype=mylinetype)
                 print('Intercept_shift=',Intercept_shift)
                 print('Intercept=',Intercept)

                 if QNRMSE<=.5:
                     
                     fit_table=pd.DataFrame(np.column_stack((new_x_data[twentyfive_index:seventyfive_index],
                                                             predicted_y_data.iloc[twentyfive_index:seventyfive_index])),
                                                            columns=["Stim_amp_pA","Frequency_Hz"])
                     my_plot+=geom_line(fit_table,aes(x=fit_table["Stim_amp_pA"],y=fit_table['Frequency_Hz']),color='green')
                     my_plot+=geom_abline(aes(intercept=Intercept,slope=Gain))
                     Threshold_table=pd.DataFrame({'Stim_amp_pA':[Threshold],'Frequency_Hz':[0]})
                     my_plot+=geom_point(Threshold_table,aes(x=Threshold_table["Stim_amp_pA"],y=Threshold_table["Frequency_Hz"]),color='green')
                     if Saturation!=np.nan:
                         my_plot+=geom_abline(aes(intercept=Saturation,slope=0))
                     
                  #my_plot+=xlab(str("Stim_amp_pA_id: "+str_cell_id))
                 print(my_plot)
                  
         

         
         return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
         
         
    except(StopIteration):
         obs='Error_Iteration'
         best_Amplitude=np.nan
         best_Half_cst=np.nan
         best_Hill_coef=np.nan
         QNRMSE=np.nan
         x_shift=np.nan
         Gain=np.nan
         Threshold=np.nan
         Saturation=np.nan
         return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
         
    except (ValueError):
          obs='Error_Value'
          best_Amplitude=np.nan
          best_Half_cst=np.nan
          best_Hill_coef=np.nan
          QNRMSE=np.nan
          x_shift=np.nan
          Gain=np.nan
          Threshold=np.nan
          Saturation=np.nan
          return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
     
    except (RuntimeError):
         obs='Error_Runtime'
         best_Amplitude=np.nan
         best_Half_cst=np.nan
         best_Hill_coef=np.nan
         QNRMSE=np.nan
         x_shift=np.nan
         Gain=np.nan
         Threshold=np.nan
         Saturation=np.nan
         return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
     
    except (TypeError):
         obs='Error_Type'
         best_Amplitude=np.nan
         best_Half_cst=np.nan
         best_Hill_coef=np.nan
         QNRMSE=np.nan
         x_shift=np.nan
         Gain=np.nan
         Threshold=np.nan
         Saturation=np.nan
         return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
     
        
def extract_inst_freq_table(SF_table, response_time):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    Cell_id : int
        Cellcell id.
    species_sweep_stim_table : DataFrame
        Coming from create_species_sweeps_stim_table function.

    Returns
    -------
    inst_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''
    
    maximum_nb_interval =0
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        stim_start_time=SF_table.loc[current_sweep,'Stim_start_s']
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(SF_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])
        
        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes
        
    new_columns=["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]

    for new_col in new_columns:
        SF_table[new_col]=np.nan


        
    for current_sweep in sweep_list:

        stim_amplitude=SF_table.loc[current_sweep,'Stim_amp_pA']
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(SF_table.loc[current_sweep,'Stim_start_s']+response_time)])
        spike_time_list=np.array(df.loc[:,'Time_s'])
        
        # Put a minimum number of spikes to compute adaptation
        if len(spike_time_list) >2:
            for current_spike_time_index in range(1,len(spike_time_list)):
                current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])
                if current_inst_frequency>1:
                    print("current_inst_freq=",current_inst_frequency)
                    print('index=',current_spike_time_index)
                    print('spike_time_i=',spike_time_list[current_spike_time_index])
                    print('spike_time_i-1=',spike_time_list[current_spike_time_index-1])
                    print('sweep=',current_sweep)
                    print(spike_time_list)
                SF_table.loc[current_sweep,str('Interval_'+str(current_spike_time_index))]=current_inst_frequency
                
            SF_table.loc[current_sweep,'Interval_1':]/=SF_table.loc[current_sweep,'Interval_1']
    # inst_freq_table = inst_freq_table.sort_values(by=["Cell_id", 'stim_amplitude_pA'])
    # inst_freq_table['Cell_id']=pd.Categorical(inst_freq_table['Cell_id'])
    
    interval_freq_table=pd.DataFrame(columns=['Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    for col in range(5,(SF_table.shape[1])):
        for line in range(SF_table.shape[0]):
            if isnull_table.iloc[line,col] == False:
                new_line=pd.Series([int(col-4), # Interval#
                                    SF_table.iloc[line,col], # Instantaneous frequency
                                    np.float64(SF_table.iloc[line,3]), # Stimulus amplitude
                                    SF_table.iloc[line,0]],# Sweep id
                                   index=['Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
                interval_freq_table=interval_freq_table.append(new_line,ignore_index=True)
   
    return interval_freq_table


def exponential_decay_function(x,A,B,C):
    '''
    Parameters
    ----------
    x : Array
        interspike interval index array.
    A: flt
        initial instantanous frequency .
    B : flt
        Adaptation index constant.
    C : flt
        intantaneous frequency limit.

    Returns
    -------
    y : array
        Modelled instantanous frequency.

    '''
    y=A*np.exp(-(x-1)/B)+C
    
    return y


def fit_adaptation_curve(interval_frequency_table,do_plot=False):
    '''
    Parameters
    ----------
    interval_frequency_table : DataFrame
        Comming from table_to_fit function.

    Returns
    -------
    my_plot : ggplot
        
    starting_freq : flt
        estimated initial instantanous frequency.
    adapt_cst : flt
        Adaptation index constant.
    limit_freq : flt
        intantaneous frequency limit.
    pcov_overall : 2-D array
        The estimated covariance of popt

    '''
    
    try:
        
        if interval_frequency_table.shape[0]==0:
            obs='Not_enough_spike'

            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            return obs,best_A,best_B,best_C,RMSE
        x_data=interval_frequency_table.loc[:,'Interval']
        y_data=interval_frequency_table.loc[:,'Normalized_Inst_frequency']
        print(interval_frequency_table)
        
        median_table=interval_frequency_table.groupby(by=["Interval"],dropna=True).median()
        median_table["Count_weights"]=pd.DataFrame(interval_frequency_table.groupby(by=["Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
        median_table["Interval"]=median_table.index
        median_table["Interval"]=np.float64(median_table["Interval"])  
        

        try: #find index of the first value lower than median Inst_freq (median or mean just between first and lowest values)
            lower_index=next(x for x, val in enumerate(median_table["Normalized_Inst_frequency"]) if val<((min(median_table["Normalized_Inst_frequency"])+median_table["Normalized_Inst_frequency"][1])/2))
        except(StopIteration):
            lower_index=np.inf
            
        try: #find index of the first value higher than median Inst_freq (median or mean just between highest and first values)
            higher_index= next(x for x, val in enumerate(median_table["Normalized_Inst_frequency"]) if val>((max(median_table["Normalized_Inst_frequency"])+median_table["Normalized_Inst_frequency"][1])/2))
        except(StopIteration):
            higher_index=np.inf


        if lower_index<higher_index: # in this case we consider that the frequency is globally decreasing
            med_index=lower_index
            initial_amplitude=1
            
            initial_decay_value_index=next(x for x, val in enumerate(median_table["Normalized_Inst_frequency"]) if val<(min(median_table["Normalized_Inst_frequency"])+(median_table["Normalized_Inst_frequency"][1]-min(median_table["Normalized_Inst_frequency"]))*(1/np.exp(1))))
            initial_decay_value=median_table["Interval"][initial_decay_value_index]
            
        else: #in this case, we consider the response is generally increasing
            med_index=higher_index
            initial_amplitude=-1
            initial_decay_value_index=next(x for x, val in enumerate(median_table["Normalized_Inst_frequency"]) if val>(median_table["Normalized_Inst_frequency"][1])*(1-(1/np.exp(1))))
            initial_decay_value=median_table["Interval"][initial_decay_value_index]
            
        
        
        
        decayModel=Model(exponential_decay_function)
        
        decay_parameters=Parameters()

        decay_parameters.add("A",value=median_table["Normalized_Inst_frequency"][1])
        decay_parameters.add('B',value=initial_decay_value,min=0)
        decay_parameters.add('C',value=median_table["Normalized_Inst_frequency"][max(median_table["Interval"])])
        
       
        result = decayModel.fit(y_data, decay_parameters, x=x_data)

        best_A=result.best_values['A']
        best_B=result.best_values['B']
        best_C=result.best_values['C']
        
        pred=exponential_decay_function(x_data,best_A,best_B,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)
        
        A_norm=best_A/(best_A+best_C)
        C_norm=best_C/(best_A+best_C)
        interval_range=np.arange(1,max(median_table["Interval"])+1,.1)

        simulation=exponential_decay_function(interval_range,best_A,best_B,best_C)
        norm_simulation=exponential_decay_function(interval_range,A_norm,best_B,C_norm)
        sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Interval","Normalized_Inst_frequency"])
        norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Interval","Normalized_Inst_frequency"])
        
        
        
        my_plot=np.nan
        if do_plot==True:
            
            my_plot=ggplot(interval_frequency_table,aes(x=interval_frequency_table["Interval"],y=interval_frequency_table["Normalized_Inst_frequency"]))+geom_point(aes(color=interval_frequency_table["Stimulus_amp_pA"]))
            
            my_plot=my_plot+geom_point(median_table,aes(x='Interval',y='Normalized_Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
            my_plot=my_plot+geom_line(sim_table,aes(x='Interval',y='Normalized_Inst_frequency'),color='black')
            my_plot=my_plot+geom_line(norm_sim_table,aes(x='Interval',y='Normalized_Inst_frequency'),color="green")

            print(my_plot)


        obs='--'
        return obs,best_A,best_B,best_C,RMSE
    
    except (StopIteration):
        obs='Error_Iteration'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except (ValueError):
        obs='Error_Value'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except(TypeError):
        obs='Error_Type'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
     

