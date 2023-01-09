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
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel,GaussianModel,QuadraticModel
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
import Data_treatment as data_treat
import Electrophy_treatment as ephys_treat

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

def get_stim_freq_table(original_SF_table, original_cell_sweep_info_table,response_time):
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    stim_freq_table=cell_sweep_info_table.copy()
    stim_freq_table['Frequency_Hz']=0
    for current_sweep in sweep_list:
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        stim_freq_table.loc[current_sweep,'Frequency_Hz']=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])/response_time

    stim_freq_table=stim_freq_table.loc[:,['Sweep', 'Stim_amp_pA', 'Frequency_Hz']]
    return stim_freq_table


def normalized_root_mean_squared_error(true, pred,pred_extended):
    #Normalization by the interquartile range
    squared_error = np.square((true - pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / true.size)
    Q1=np.percentile(pred_extended,25)
    Q3=np.percentile(pred_extended,75)
    print('rmse=',rmse)
    nrmse_loss = rmse/(Q3-Q1)
    return nrmse_loss

def hill_function(x, Amplitude,Hill_coef,Half_cst):
    return Amplitude*((x**(Hill_coef))/((Half_cst**Hill_coef)+(x**(Hill_coef))))


def fit_IO_curve(original_stimulus_frequency_table,do_plot=False):
    
    try:
         stimulus_frequency_table=original_stimulus_frequency_table.copy()
         stimulus_frequency_table=stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
         x_shift=abs(min(stimulus_frequency_table['Stim_amp_pA']))
         stimulus_frequency_table.loc[:,'Stim_amp_pA']=stimulus_frequency_table.loc[:,'Stim_amp_pA']+x_shift
         stimulus_frequency_table=stimulus_frequency_table.reset_index(drop=True)
         #stimulus_frequency_table.index=[x for x in range(stimulus_frequency_table.shape[0]+1)]

         x_data=stimulus_frequency_table.loc[:,'Stim_amp_pA']
         y_data=stimulus_frequency_table.loc[:,"Frequency_Hz"]

         
         #get initial estimate of parameters for single sigmoid fit
         
         if len(np.flatnonzero(y_data))>0:
             without_zero_index=np.flatnonzero(y_data)[0]
         else:
             without_zero_index=y_data.iloc[0]
         
         median_firing_rate_index=np.argmax(y_data>np.median(y_data.iloc[without_zero_index:]))

         #Get the stimulus amplitude correspondingto the median non-zero firing rate
         x0=x_data.iloc[median_firing_rate_index]

         
         max_freq_step_index=np.argmax(y_data.diff())
         freq_step_array=y_data.diff()
         stim_step_array=x_data.diff()
         normalized_step_array=freq_step_array/stim_step_array
         max_step=np.nanmax(normalized_step_array)
         max_freq_step=freq_step_array[max_freq_step_index]
         max_stim_step=stim_step_array[max_freq_step_index]
         normalized_step=max_freq_step/max_stim_step
         
         sub_x_data=x_data.iloc[without_zero_index:]
         sub_y_data=y_data.iloc[without_zero_index:]
         print('Normalisez_step=',normalized_step)
         # if normalized_step>=1.5:
         #     obs='Max_step_too_high'
         #     best_Amplitude=np.nan
         #     best_Half_cst=np.nan
         #     best_Hill_coef=np.nan
         #     QNRMSE=np.nan
         #     x_shift=np.nan
         #     Gain=np.nan
         #     Threshold=np.nan
         #     Saturation=np.nan
             
         
         # else: #If the highest step is not too high, try to fit 
         Gain=np.nan
         Threshold=np.nan
         Saturation=np.nan
         hillModel=Model(hill_function)
         hill_parameters=Parameters()
         hill_parameters.add("Half_cst",value=x0,min=0)
         hill_parameters.add('Hill_coef',value=max_step,min=0)
         hill_parameters.add('Amplitude',value=max(y_data))
         print(hill_parameters)
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
            #print('Intercept_shift=',Intercept_shift)
            #print('Intercept=',Intercept)
   
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
     
    # except (TypeError):
    #      obs='Error_Type'
    #      best_Amplitude=np.nan
    #      best_Half_cst=np.nan
    #      best_Hill_coef=np.nan
    #      QNRMSE=np.nan
    #      x_shift=np.nan
    #      Gain=np.nan
    #      Threshold=np.nan
    #      Saturation=np.nan
    #      return obs,best_Amplitude,best_Hill_coef,best_Half_cst,QNRMSE,x_shift,Gain,Threshold,Saturation
     
        
def extract_inst_freq_table(original_SF_table, original_cell_sweep_info_table,response_time):
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
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes

            
            
    new_columns=["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]

    
    SF_table[new_columns] = (np.nan*maximum_nb_interval)
    
    for current_sweep in sweep_list:


        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)])
        spike_time_list=np.array(df.loc[:,'Time_s'])
        
        # Put a minimum number of spikes to compute adaptation
        if len(spike_time_list) >2:
            for current_spike_time_index in range(1,len(spike_time_list)):
                current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])

                SF_table.loc[current_sweep,str('Interval_'+str(current_spike_time_index))]=current_inst_frequency

                
        SF_table.loc[current_sweep,'Interval_1':]/=SF_table.loc[current_sweep,'Interval_1']
    #return (SF_table)
    interval_freq_table=pd.DataFrame(columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    isnull_table.columns=SF_table.columns
    isnull_table.index=SF_table.index
    
    for interval,col in enumerate(new_columns):
        for line in sweep_list:
            if isnull_table.loc[line,col] == False:

                new_line=pd.Series([int(interval)+1, # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line],# Sweep id
                                   index=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
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


def fit_adaptation_curve(original_interval_frequency_table,do_plot=False):
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
        interval_frequency_table=original_interval_frequency_table.copy()
        if interval_frequency_table.shape[0]==0:
            obs='Not_enough_interval'

            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            Adaptation_fit_table=np.nan
            return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
        
        sweep_list=interval_frequency_table['Sweep'].unique()
        Adaptation_fit_table=pd.DataFrame(columns=["Sweep","Stim_amp_pA","A",'B','C','Nb_of_spikes','RMSE'])
        fit_table=pd.DataFrame(columns=['Spike_Interval','Normalized_Inst_frequency','Sweep','Nb_of_spikes','Stim_amp_pA'])
        
        interval_range=np.arange(1,max(interval_frequency_table["Spike_Interval"])+1,.1)

        for current_sweep in sweep_list:

            sub_interval_frequency_table=interval_frequency_table[interval_frequency_table['Sweep']==current_sweep]
            sub_interval_frequency_table.index=sub_interval_frequency_table['Spike_Interval']
            stim_amplitude=sub_interval_frequency_table.iloc[0,2]
            
            sub_x_data=sub_interval_frequency_table.loc[:,'Spike_Interval']
            sub_y_data=sub_interval_frequency_table.loc[:,'Normalized_Inst_frequency']
            
            if sub_x_data.shape[0]<4:
                # new_line=pd.Series([current_sweep,stim_amplitude,np.nan,np.nan,np.nan,sub_x_data.shape[0]+1],
                #                    index=["Sweep","Stim_amp_pA","A",'B','C','Nb_of_spikes'])
                # Adaptation_fit_table=Adaptation_fit_table.append(new_line,ignore_index=True)
                
                continue
                
                
       
            
            sub_y_data=sub_y_data.reset_index(drop=True)
            sub_x_data=sub_x_data.reset_index(drop=True)

            y_delta=sub_y_data.iloc[-1]-sub_y_data.iloc[0]
            y_delta_two_third=sub_y_data.iloc[0]-.66*y_delta
            
            initial_time_cst_guess_idx=np.argmin(abs(sub_y_data - y_delta_two_third))
            initial_time_cst_guess=sub_x_data[initial_time_cst_guess_idx]
            
            
            
            
            
            initial_A=(sub_y_data.iloc[0]-sub_y_data.iloc[-1])/np.exp(-sub_x_data.iloc[0]/initial_time_cst_guess)

            decayModel=Model(exponential_decay_function)
            
            decay_parameters=Parameters()
    
            decay_parameters.add("A",value=initial_A)

            decay_parameters.add('B',value=initial_time_cst_guess,min=1e-9)
            decay_parameters.add('C',value=sub_interval_frequency_table["Normalized_Inst_frequency"][max(sub_interval_frequency_table["Spike_Interval"])])
            
            result = decayModel.fit(sub_y_data, decay_parameters, x=sub_x_data)
    
            current_A=result.best_values['A']
            current_B=result.best_values['B']
            current_C=result.best_values['C']
            
            pred=exponential_decay_function(sub_x_data,current_A,current_B,current_C)
            squared_error = np.square((sub_y_data - pred))
            sum_squared_error = np.sum(squared_error)
            current_RMSE = np.sqrt(sum_squared_error / sub_y_data.size)
            
            new_line=pd.Series([current_sweep,stim_amplitude,current_A,current_B,current_C,sub_interval_frequency_table.shape[0]+1,current_RMSE],
                               index=["Sweep","Stim_amp_pA","A",'B','C','Nb_of_spikes','RMSE'])
            Adaptation_fit_table=Adaptation_fit_table.append(new_line,ignore_index=True)

            current_simulation=exponential_decay_function(interval_range,current_A, current_B, current_C)

            current_simulation_table=pd.DataFrame(np.column_stack((interval_range,current_simulation)),
                                                  columns=["Spike_Interval","Normalized_Inst_frequency"])
            current_simulation_table['Sweep']=current_sweep
            current_simulation_table['Nb_of_spikes']=int(sub_interval_frequency_table.shape[0]+1)
            current_simulation_table["Stim_amp_pA"]=stim_amplitude
            fit_table=fit_table.append(current_simulation_table,ignore_index=True)
            if do_plot:
                current_plot=ggplot()+geom_point(sub_interval_frequency_table,aes(x="Spike_Interval",y="Normalized_Inst_frequency"))+geom_line(current_simulation_table,aes(x="Spike_Interval",y="Normalized_Inst_frequency"))+xlab(str('Spike_interval_Sweep_'+str(current_sweep)))
                print(current_plot)

            
        
        x_data=interval_frequency_table.loc[:,'Spike_Interval']
        y_data=interval_frequency_table.loc[:,'Normalized_Inst_frequency']
        Adaptation_fit_table=Adaptation_fit_table[Adaptation_fit_table['Nb_of_spikes']>=4]
        
        if Adaptation_fit_table.shape[0]==0:
            obs='Not_enough_interval'
            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            Adaptation_fit_table=np.nan
            return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
        
        elif Adaptation_fit_table[Adaptation_fit_table['A']<0].shape[0]/Adaptation_fit_table.shape[0]>.2:
            obs='Too_many_accelerating_sweeps'
            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            Adaptation_fit_table=np.nan
            return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
        best_A=np.average(Adaptation_fit_table['A'],weights=Adaptation_fit_table['Nb_of_spikes'])
        best_B=np.average(Adaptation_fit_table['B'],weights=Adaptation_fit_table['Nb_of_spikes'])
        best_C=np.average(Adaptation_fit_table['C'],weights=Adaptation_fit_table['Nb_of_spikes'])
        
        pred=exponential_decay_function(x_data,best_A,best_B,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)
        
        simulation=exponential_decay_function(interval_range,best_A,best_B,best_C)

        sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])
        #norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Interval","Normalized_Inst_frequency"])
        
        if do_plot==True:
            
            my_plot=ggplot(interval_frequency_table,aes(x=interval_frequency_table["Spike_Interval"],y=interval_frequency_table["Normalized_Inst_frequency"]))+geom_point(aes(color=interval_frequency_table["Stimulus_amp_pA"]))
            my_plot=my_plot+geom_line(fit_table,aes(x=fit_table['Spike_Interval'],y=fit_table['Normalized_Inst_frequency'],group=fit_table['Sweep'],color='Stim_amp_pA',alpha="Nb_of_spikes"))
            #my_plot=my_plot+geom_point(median_table,aes(x='Interval',y='Normalized_Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
            my_plot=my_plot+geom_line(sim_table,aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color='red')
            #my_plot=my_plot+geom_line(norm_sim_table,aes(x='Interval',y='Normalized_Inst_frequency'),color="green")

            print(my_plot)


        obs='--'
        return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
    
    except (StopIteration):
        obs='Error_Iteration'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        Adaptation_fit_table=np.nan
        return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
    # except (ValueError):
    #     obs='Error_Value'
    #     best_A=np.nan
    #     best_B=np.nan
    #     best_C=np.nan
    #     RMSE=np.nan
    #     return obs,best_A,best_B,best_C,RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        Adaptation_fit_table=np.nan
        return obs,best_A,best_B,best_C,RMSE,Adaptation_fit_table
   
     

def time_cst_model(x,A,tau,C):
    y=A*np.exp(-(x)/tau)+C
    return y

def fit_membrane_trace (original_time_membrane_table,start_time,end_time,do_plot=False):
    try:
        time_membrane_table=original_time_membrane_table.copy()
        x_data=np.array(time_membrane_table.loc[:,'Time_s'])
        y_data=np.array(time_membrane_table.loc[:,"Membrane_potential_mV"])
        start_idx = np.argmin(abs(x_data - start_time))
        end_idx = np.argmin(abs(x_data - end_time))
        
        
        y_delta=y_data[-1]-y_data[0]
        y_delta_two_third=y_data[0]-.66*y_delta
            
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        initial_time_cst_guess=x_data[initial_time_cst_guess_idx]
            
            
            
            
            
        initial_A=(y_data[0]-y_data[-1])/np.exp(-x_data[0]/initial_time_cst_guess)

       
        # membrane_baseline=np.mean(y_data[:start_idx])
        # mid_idx=int((end_idx+start_idx)/2)

        # membrane_SS=np.mean(y_data[mid_idx:end_idx])
        
        # membrane_delta=membrane_SS-membrane_baseline

        # initial_potential_time_cst=membrane_baseline+(1-(1/np.exp(1)))*membrane_delta

        # initial_potential_time_cst_idx=np.argmin(abs(y_data[start_idx:end_idx] - initial_potential_time_cst))+start_idx
        # initial_time_cst=x_data[initial_potential_time_cst_idx]-start_time
        

        
        # initial_A=(y_data[start_idx]-membrane_SS)/np.exp(-x_data[start_idx]/initial_time_cst)
        
        time_cst_Model=Model(time_cst_model)
    
        time_cst_parameters=Parameters()

        time_cst_parameters.add("A",value=initial_A)
        time_cst_parameters.add('tau',value=initial_time_cst_guess)
        time_cst_parameters.add('C',value=y_data[-1])

        result = time_cst_Model.fit(y_data[start_idx:end_idx], time_cst_parameters, x=x_data[start_idx:end_idx])
        best_A=result.best_values['A']
        best_tau=result.best_values['tau']
        best_C=result.best_values['C']
        
        
        if do_plot==True:
            simulation=time_cst_model(x_data[start_idx:end_idx],best_A,best_tau,best_C)
            sim_table=pd.DataFrame(np.column_stack((x_data[start_idx:end_idx],simulation)),columns=["Time_s","Membrane_potential_mV"])
            
            my_plot=ggplot(time_membrane_table,aes(x=time_membrane_table["Time_s"],y=time_membrane_table["Membrane_potential_mV"]))+geom_line(color='blue')#+xlim((start_time-.1),(end_time+.1))
            
            
            my_plot=my_plot+geom_line(sim_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red')
            
    
            print(my_plot)
            
        return best_A,best_tau,best_C
    except (ValueError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        return best_A,best_tau,best_C
    
# def double_time_cst_model(x,A,tau,C,D):
#     y=A*np.exp(-(x)/tau)+C+D
#     return y

# def Heaviside_cst_exp_function(x, stim_start, stim_end,baseline,A,tau,C):
#     """Heaviside step function."""
    
#     if stim_end<=min(x):
#         o=np.empty(x.size);o.fill(C)
#         return o
    
#     elif stim_start>=max(x):
#         o=np.empty(x.size);o.fill(baseline)
#         return o
    
#     else:
#         o=np.empty(x.size);o.fill(baseline)
        
        
#         start_index = max(np.where( x < stim_start)[0])

#         end_index=max(np.where( x < stim_end)[0])
#         o[:start_index]=baseline
#         o[start_index:end_index] = time_cst_model(x[start_index:end_index],A,tau,C)
    
#         return o
    
# def fit_membrane_time_cst (original_time_membrane_table,start_time,end_time,do_plot=False):
#     try:
#         time_membrane_table=original_time_membrane_table.copy()
#         x_data=np.array(time_membrane_table.loc[:,'Time_s'])
#         y_data=np.array(time_membrane_table.loc[:,"Membrane_potential_mV"])
#         start_idx = np.argmin(abs(x_data - start_time))
#         end_idx = np.argmin(abs(x_data - end_time))
       
#         membrane_baseline=np.mean(y_data[:start_idx])
#         mid_idx=int((end_idx+start_idx)/2)

#         membrane_SS=np.mean(y_data[mid_idx:end_idx])
        
#         membrane_delta=membrane_SS-membrane_baseline

#         initial_potential_time_cst=membrane_baseline+(1-(1/np.exp(1)))*membrane_delta

#         initial_potential_time_cst_idx=np.argmin(abs(y_data[start_idx:end_idx] - initial_potential_time_cst))+start_idx
#         initial_time_cst=x_data[initial_potential_time_cst_idx]-start_time
        

        
#         initial_A=(y_data[start_idx]-membrane_SS)/np.exp(-x_data[start_idx]/initial_time_cst)
        
        
        
        
#         double_step_model=Model(Heaviside_cst_exp_function)
#         #double_step_model_pars=double_step_model.make_params(stim_start=start_time,stim_end=end_time,baseline=membrane_baseline,A=initial_A,tau=initial_time_cst,C=membrane_SS)
#         double_step_model_pars=Parameters()
        
#         double_step_model_pars.add('stim_start',value=start_time)
#         double_step_model_pars.add('stim_end',value=end_time,vary=False)
#         double_step_model_pars.add('baseline',value=membrane_baseline,vary=False)
#         double_step_model_pars.add('A',value=initial_A)
#         double_step_model_pars.add('tau',value=initial_time_cst)
#         double_step_model_pars.add('C',value=membrane_SS)
#         print('okovf')

#         # print(double_step_model.param_hints)
#         #init=double_step_model.eval( pars, x=x_data[:end_idx])  
#         #return(init)
#         double_step_out=double_step_model.fit(y_data[:end_idx], double_step_model_pars, x=x_data[:end_idx])        
#         print('ijr')
#         print(double_step_out.best_values)
#         best_A=double_step_out.best_values['A']
#         best_tau=double_step_out.best_values['tau']
#         best_C=double_step_out.best_values['C']
#         best_stim_start=double_step_out.best_values['stim_start']
#         best_stim_end=double_step_out.best_values['stim_end']
#         best_baseline=double_step_out.best_values["baseline"]
        
        
#         if do_plot==True:
#             simulation=Heaviside_cst_exp_function(x_data[:end_idx],best_stim_start,best_stim_end,best_baseline,best_A,best_tau,best_C)
#             sim_table=pd.DataFrame(np.column_stack((x_data[:end_idx],simulation)),columns=["Time_s","Membrane_potential_mV"])
            
#             my_plot=ggplot(time_membrane_table,aes(x=time_membrane_table["Time_s"],y=time_membrane_table["Membrane_potential_mV"]))+geom_line(color='blue')#+xlim((start_time-.1),(end_time+.1))
            
            
#             my_plot=my_plot+geom_line(sim_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red')
            
    
#             print(my_plot)
            
#         return best_A,best_tau,best_C
#     except (NameError):
#         best_A=np.nan
#         best_tau=np.nan
#         best_C=np.nan
#         return best_A,best_tau,best_C
    

def fit_second_order_poly(original_fit_table,do_plot=False):
    fit_table=original_fit_table.copy()
    x_data=np.array(fit_table.loc[:,'Time_s'])
    y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
    
    poly_model=QuadraticModel()
    pars = poly_model.guess(y_data, x=x_data)
    out = poly_model.fit(y_data, pars, x=x_data)
    
    a=out.best_values["a"]
    b=out.best_values["b"]
    c=out.best_values["c"]
    
    pred=a*((x_data)**2)+b*x_data+c
    squared_error = np.square((y_data - pred))
    sum_squared_error = np.sum(squared_error)
    RMSE_poly = np.sqrt(sum_squared_error / y_data.size)
    
    
    
    fit_table['Data']='Original_Data'
    simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
    simulation_table['Data']='Fit_Poly_Data'
   # fit_table=pd.concat([fit_table, simulation_table], axis=0)
    if do_plot:
        
        my_plot=ggplot(fit_table,aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+geom_line()+ylim(-72,-70)
        my_plot+=geom_line(simulation_table,aes(x="Time_s",y="Membrane_potential_mV",color='Data'))
        print(my_plot)
    return a,b,c,RMSE_poly
    

def fit_exponential(original_fit_table,do_plot=False):
    fit_table=original_fit_table.copy()
    x_data=np.array(fit_table.loc[:,'Time_s'])
    y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
    expo_model=ExponentialModel()
    pars = expo_model.guess(y_data, x=x_data)
    out = expo_model.fit(y_data, pars, x=x_data)
    
    A=out.best_values["amplitude"]
    tau=out.best_values["decay"]

    
    pred=A*np.exp(-(x_data)/tau)
    squared_error = np.square((y_data - pred))
    sum_squared_error = np.sum(squared_error)
    RMSE_expo = np.sqrt(sum_squared_error / y_data.size)
    
    
    fit_table['Data']='Original_Data'
    simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
    simulation_table['Data']='Fit_Expo_Data'
    fit_table=pd.concat([fit_table, simulation_table], axis=0)
    if do_plot:
        
        my_plot=ggplot(fit_table,aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+geom_line()
        print(my_plot)
    return A,tau,RMSE_expo

    
    
    
    
    
    
    
    

