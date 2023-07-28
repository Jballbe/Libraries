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

import time
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle,geom_segment

import scipy
from scipy.stats import linregress
from scipy import optimize
import random

import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import warnings
import pandas
import os
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel,GaussianModel,QuadraticModel,PolynomialModel,SkewedGaussianModel
from lmfit import Parameters, Minimizer,fit_report
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual,scale_shape_manual
from plotnine.labels import xlab
from plotnine.coords import coord_cartesian
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.interpolate import splrep, splev
from scipy.special import erfc, erf
from scipy.misc import derivative
from scipy import io
import math
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

def get_stim_freq_table(original_SF_table, original_cell_sweep_info_table,sweep_QC_table,response_duration, response_based="Time_based"):
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    stim_freq_table=cell_sweep_info_table.copy()
    stim_freq_table['Frequency_Hz']=0

    for current_sweep in sweep_list:
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'].copy())
        df=df[df['Feature']=='Upstroke']
        
        if response_based == 'Time_based':
            stim_freq_table.loc[current_sweep,'Frequency_Hz']=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_duration)].shape[0])/response_duration
           
        elif response_based == 'Index_based':
            df=df.sort_values(by=["Time_s"])
            df=df.iloc[:response_duration,:]
            if df.shape[0]==0:
                stim_freq_table.loc[current_sweep,'Frequency_Hz']=0
            else:
                last_time=np.nanmax(df['Time_s'])
                stim_freq_table.loc[current_sweep,'Frequency_Hz'] = df.shape[0]/(last_time-cell_sweep_info_table.loc[current_sweep,'Stim_start_s'])
            
        elif response_based == 'Interval_based':
            df=df.sort_values(by=["Time_s"])
            
            if df.shape[0]<int(response_duration+1):
                stim_freq_table.loc[current_sweep,'Frequency_Hz']=0
            else:
                spike_time=np.array(df['Time_s'])
                stim_freq_table.loc[current_sweep,'Frequency_Hz']=1/(spike_time[response_duration]-spike_time[response_duration-1])
            
            
            
    stim_freq_table=stim_freq_table.loc[:,['Sweep', 'Stim_amp_pA', 'Frequency_Hz']]
    stim_freq_table=pd.merge(stim_freq_table,sweep_QC_table,on='Sweep')
    return stim_freq_table


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

def hill_function_derivative(x, Amplitude,Hill_coef,Half_cst):
    return Amplitude*((Hill_coef*(x**(Hill_coef-1))*(Half_cst**Hill_coef))/((Half_cst**Hill_coef)+(x**(Hill_coef)))**2)

def sigmoid_function(x,x0,sigma):
    return (1-(1/(1+np.exp((x-x0)/sigma))))

def hill_sigmoid_function(x, Amplitude,Hill_coef,Half_cst,x0,sigma):
    return Amplitude*((x**(Hill_coef))/((Half_cst**Hill_coef)+(x**(Hill_coef)))) * (1-(1/(1+np.exp((x-x0)/sigma))))

def hill_function_new (x, x0, Hill_coef, Half_cst):
    
    y=np.empty(x.size)
    
    
    
    
    if len(max(np.where( x <= x0))) !=0:


        x0_index = max(np.where( x <= x0)[0])+1
        
    
        y[:x0_index]=0.

        y[x0_index:] =(((x[x0_index:]-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x[x0_index:]-x0)**(Hill_coef))))

    else:

        y = (((x-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x-x0)**(Hill_coef))))
        
    return y

def sigmoid_amplitude_function(x,A,x0,sigma):
    
    return A*(1-(1/(1+np.exp((x-x0)/sigma))))

def fit_IO_relationship(original_stimulus_frequency_table,do_plot=False):
    
    
    scale_dict = {'3rd_order_poly' : 'black',
                  '2nd_order_poly' : 'black',
                  'Trimmed_asc_seg' : 'pink',
                  "Trimmed_desc_seg" : "red",
                  'Ascending_Sigmoid' : 'blue',
                  "Descending_Sigmoid"  : 'orange',
                  'Amplitude' : 'yellow',
                  "Asc_Hill_Desc_Sigmoid": "red",
                  "First_Hill_Fit" : 'green',
                  'Hill_Sigmoid_Fit' : 'green',
                  'Sigmoid_fit' : "blue",
                  'Hill_Fit' : 'green',
                  True : 'o',
                  False : 's'
                  
                  }
    
    best_value_Amp=np.nan
    best_H_Hill_coef=np.nan
    best_H_Half_cst=np.nan
    best_H_x0=np.nan
    best_value_x0_desc=np.nan
    best_value_sigma_desc=np.nan
    best_QNRMSE=np.nan
    x_shift=np.nan
    Gain=np.nan
    Threshold=np.nan
    Saturation_frequency=np.nan
    Saturation_stimulation=np.nan
    
    
    
    original_data_table =  original_stimulus_frequency_table.copy()
    original_data_shifted = original_data_table.copy()
    x_shift=abs(min(original_data_shifted['Stim_amp_pA']))
    #original_data_shifted.loc[:,'Stim_amp_pA']=original_data_shifted.loc[:,'Stim_amp_pA']+x_shift
    
    if 'Passed_QC' in original_data_shifted.columns:
        original_data_shifted=original_data_shifted[original_data_shifted['Passed_QC']==True]
    
    
   
    
    original_data_shifted = original_data_shifted.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    original_x_data = np.array(original_data_shifted['Stim_amp_pA'])
    original_y_data = np.array(original_data_shifted['Frequency_Hz'])
    extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
    
    first_stimulus_frequency_table=original_data_shifted.copy()
    first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    
    ### 0 - Trim Data
    response_threshold = (np.nanmax(first_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(first_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
    
    for elt in range(first_stimulus_frequency_table.shape[0]):
        if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
            first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
        else:
            break
    
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
    for elt in range(first_stimulus_frequency_table.shape[0]):
        if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
            first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
        else:
            break
    
    ### 0 - end
    
    ### 1 - Try to fit a 3rd order polynomial
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    trimmed_x_data=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']
    trimmed_y_data=first_stimulus_frequency_table.loc[:,"Frequency_Hz"]
    extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))

    if len(trimmed_y_data)>=4:
        third_order_poly_model = PolynomialModel(degree = 3)

        pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)

        best_c0 = third_order_poly_model_results.best_values['c0']
        best_c1 = third_order_poly_model_results.best_values['c1']
        best_c2 = third_order_poly_model_results.best_values['c2']
        best_c3 = third_order_poly_model_results.best_values['c3']
        
        delta_3rd_order_poly = 4*(best_c2**2)-4*3*best_c3*best_c1 #check if 3rd order polynomial reaches local max/minimum
        


        extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        
        trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_3rd_poly_model})
        trimmed_3rd_poly_table['Legend']='3rd_order_poly'
        
        if do_plot:
            
            polynomial_plot = ggplot()
            polynomial_plot += geom_point(original_data_shifted,aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
            polynomial_plot += geom_point(first_stimulus_frequency_table,aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
            polynomial_plot += geom_line(trimmed_3rd_poly_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += ggtitle('3rd order polynomial fit to trimmed_data')
            polynomial_plot += scale_color_manual(values=scale_dict)
            polynomial_plot += scale_shape_manual(values=scale_dict)
            print(polynomial_plot)
            
        extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
        zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
        if len(zero_crossings_3rd_poly_model_freq_diff)==0: # if the derivative of the 3rd order polynomial does not change sign --> Assume there is no Descending Segment
            Descending_segment=False
            ascending_segment = trimmed_3rd_poly_table.copy()
            
        elif len(zero_crossings_3rd_poly_model_freq_diff)==1:# if the derivative of the 3rd order polynomial changes sign 1 time
            first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
            if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                ## if before the change of sign the derivatove of the 3rd order poly is positive, 
                ## then we know there is a Descending Segment after the change of sign
                Descending_segment=True
                
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                
            elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                ## if before the change of sign the derivatove of the 3rd order poly is negative, 
                ## then we know the "Descending Segment" is fitted to the beginning of the data = artifact --> there is no Descending segment after the Acsending Segment
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                
        elif len(zero_crossings_3rd_poly_model_freq_diff)==2:# if the derivative of the 3rd order polynomial changes sign 2 times
            first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
            second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
            
            
            if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                ## if before the first change of sign the derivative of the 3rd order poly is positive
                ## then we consider the Ascending Segment before the first root and the Descending Segment after the First root
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
                descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
                
                
                trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
                
                descending_linear_slope_init,descending_linear_intercept_init=fit_specimen_fi_slope(trimmed_descending_segment["Stim_amp_pA"],
                                                      trimmed_descending_segment['Frequency_Hz'])
                
                if descending_linear_slope_init<=0:
                    Descending_segment = True
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                else:
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                
                
            elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                Descending_segment=True
                ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
        
        ### 1 - end
    
    else:
        Descending_segment=False
    

    
    if Descending_segment:
        
        obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,sigmoid_x0, sigmoid_sigma,Legend = fit_Hill_Sigmoid(original_data_table, third_order_poly_model_results, ascending_segment, descending_segment, scale_dict,do_plot=do_plot)
        
        if obs == 'Hill-Sigmoid':
            feature_obs,best_QNRMSE,x_shift_second,Gain,Threshold,Saturation_frequency,Saturation_stimulation=extract_IO_features(original_data_table,x_shift,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,do_plot)

            if feature_obs == 'Hill-Sigmoid':
                return feature_obs,Amp,Hill_coef,Hill_Half_cst,Hill_x0,sigmoid_x0,sigmoid_sigma,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation
            
            else:
                Descending_segment=False
                obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend = fit_Single_Hill(original_data_table, scale_dict,do_plot=do_plot)
                sigmoid_x0=np.nan
                sigmoid_sigma = np.nan

                if obs =='Hill':
                    feature_obs,best_QNRMSE,x_shift_second,Gain,Threshold,Saturation_frequency,Saturation_stimulation=extract_IO_features(original_data_table,x_shift,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,do_plot)
                    
                else:
                    empty_array = np.empty(7)
                    empty_array[:] = np.nan
                    feature_obs,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
                
        
        else:
            Descending_segment=False
            obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend = fit_Single_Hill(original_data_table, scale_dict,do_plot=do_plot)
            sigmoid_x0=np.nan
            sigmoid_sigma = np.nan
            if obs =='Hill':
                feature_obs,best_QNRMSE,x_shift_second,Gain,Threshold,Saturation_frequency,Saturation_stimulation=extract_IO_features(original_data_table,x_shift,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,do_plot)
                
            else:
                empty_array = np.empty(7)
                empty_array[:] = np.nan
                feature_obs,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
            
    
            
    else:
        
        obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend = fit_Single_Hill(original_data_table, scale_dict,do_plot=do_plot)
        sigmoid_x0=np.nan
        sigmoid_sigma = np.nan
        if obs =='Hill':
            feature_obs,best_QNRMSE,x_shift_second,Gain,Threshold,Saturation_frequency,Saturation_stimulation=extract_IO_features(original_data_table,x_shift,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,do_plot)
        else:
            empty_array = np.empty(6)
            empty_array[:] = np.nan
           
            feature_obs=obs
            best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
        
    return feature_obs,Amp,Hill_coef,Hill_Half_cst,Hill_x0,sigmoid_x0,sigmoid_sigma,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation

    
    
    
def fit_Hill_Sigmoid(original_data_init, third_order_poly_model_results,ascending_segment,descending_segment,scale_dict,do_plot=False):
    
    try:   
        original_data=original_data_init.copy()
        

        #x_shift=abs(min(original_data['Stim_amp_pA']))
        #original_data.loc[:,'Stim_amp_pA']=original_data.loc[:,'Stim_amp_pA']+x_shift
        first_stimulus_frequency_table=original_data.copy()
        if 'Passed_QC' in first_stimulus_frequency_table.columns:
            first_stimulus_frequency_table=first_stimulus_frequency_table[first_stimulus_frequency_table['Passed_QC']==True]
        
        
       
                
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data['Stim_amp_pA'])
        original_y_data = np.array(original_data['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
        

        
    
        ### 0 - Trim Data
        response_threshold = (np.nanmax(first_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(first_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
        
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']

        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        best_c0 = third_order_poly_model_results.best_values['c0']
        best_c1 = third_order_poly_model_results.best_values['c1']
        best_c2 = third_order_poly_model_results.best_values['c2']
        best_c3 = third_order_poly_model_results.best_values['c3']
        
        extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_3rd_poly_model})
        trimmed_3rd_poly_table['Legend']='3rd_order_poly'
        
        
        
        ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
        ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
        
        descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
        descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
        
        trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
        trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
        
        
        max_freq_poly_fit = np.nanmax([np.nanmax(ascending_segment['Frequency_Hz']),np.nanmax(descending_segment['Frequency_Hz'])])
        #max_freq_poly_fit=np.nanmax(trimmed_3rd_poly_table['Frequency_Hz'])
        
        trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"

        trimmed_descending_segment.loc[trimmed_descending_segment.index,'Legend']="Trimmed_desc_seg"
        if do_plot:
            
            polynomial_plot = ggplot()
            polynomial_plot += geom_point(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
            polynomial_plot += geom_point(first_stimulus_frequency_table,aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
            polynomial_plot += geom_line(trimmed_3rd_poly_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += ggtitle('3rd order polynomial fit to trimmed_data')
            polynomial_plot += scale_color_manual(values=scale_dict)
            polynomial_plot += scale_shape_manual(values=scale_dict)
            print(polynomial_plot)
            trimmed_table=trimmed_ascending_segment.append(trimmed_descending_segment, ignore_index=True)
            polynomial_plot+=geom_line(trimmed_table,aes(x='Stim_amp_pA',y="Frequency_Hz",colour='Legend',group="Legend"))
            polynomial_plot += scale_color_manual(values=scale_dict)
            polynomial_plot += scale_shape_manual(values=scale_dict)
            #polynomial_plot+=geom_line(trimmed_descending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz"),colour='red')
            print(polynomial_plot)
        
        
        ascending_linear_slope_init,ascending_linear_intercept_init=fit_specimen_fi_slope(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        descending_linear_slope_init,descending_linear_intercept_init=fit_specimen_fi_slope(trimmed_descending_segment["Stim_amp_pA"],
                                              trimmed_descending_segment['Frequency_Hz'])
        
        if do_plot:   
            polynomial_plot+=geom_abline(slope=ascending_linear_slope_init,
                                         intercept=ascending_linear_intercept_init,
                                         colour="pink",
                                         linetype='dashed')
            
            polynomial_plot+=geom_abline(slope=descending_linear_slope_init,
                                         intercept=descending_linear_intercept_init,
                                         colour="red",
                                         linetype='dashed')
            
            print(polynomial_plot)
        
        ### 2 - Fit Double Sigmoid : Amp*Sigmoid_asc*Sigmoid_desc
        
        # Ascending Sigmoid
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        # Amplitude
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        # Amp*Ascending segment
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        # Descending segment
        descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
        descending_sigmoid_fit = Model(sigmoid_function,prefix='desc_')
        descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
        descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
        descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
        
        # Amp*Ascending_sigmoid*Descending_sigmoid
        ascending_sigmoid_fit *=descending_sigmoid_fit
        ascending_segment_fit_params+=descending_sigmoid_fit_pars
        
        # Fit model to original data
        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        # Get fit best parameters
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
        best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
        
        # Create simulation table
        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        
        if do_plot:
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data['Frequency_Hz'])/max(asc_sig_comp_table['Frequency_Hz'])
            amp_comp = double_sigmoid_comps['Amp_']
            amp_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : amp_comp,
                                        'Legend' : "Amplitude"})
            
            component_table = asc_sig_comp_table.append(amp_comp_table,ignore_index=True)
            component_plot = ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
           
            
            sigmoid_fit_plot =  ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
            sigmoid_fit_plot += geom_line(full_sigmoid_fit_table, aes(x='Stim_amp_pA',y="Frequency_Hz"))
            sigmoid_fit_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += ggtitle("Sigmoid_fit_to original_data")
            
            desc_sig_comp = double_sigmoid_comps['desc_']
            desc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : desc_sig_comp,
                                        'Legend' : "Descending_Sigmoid"})
            desc_sig_comp_table['Frequency_Hz'] = desc_sig_comp_table['Frequency_Hz']*max(original_data['Frequency_Hz'])/max(desc_sig_comp_table['Frequency_Hz'])
            component_table = component_table.append(desc_sig_comp_table,ignore_index=True)
            
            sigmoid_fit_plot += geom_line(trimmed_descending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += geom_line(trimmed_descending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        
            component_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            component_plot += ggtitle('Sigmoid_fit_components')
            component_plot += scale_color_manual(values=scale_dict)
            component_plot += scale_shape_manual(values=scale_dict)
            print(component_plot)
            
            
            sigmoid_fit_plot += scale_color_manual(values=scale_dict)
            sigmoid_fit_plot += scale_shape_manual(values=scale_dict)
            print(sigmoid_fit_plot)
            
        #Determine Hill half coef for first hill fit --> First stim_amp to elicit half max(frequency)
        half_response_index=np.argmax(original_y_data>=(0.5*max(original_y_data)))
       
        half_response_stim = original_x_data[half_response_index]
        
        # Set limit to Hill coef to prevent the parameter space exporation to go too high which breaks the fit
        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
    
       
        # Create Hill model to fit on ascending sigmoid
        first_Hill_model = Model(hill_function_new)
        first_Hill_pars = first_Hill_model.make_params()
        first_Hill_pars.add("x0",value=np.nanmin(original_x_data),min=-100)
        first_Hill_pars.add("Half_cst",value=half_response_stim,min=1e-9)
        first_Hill_pars.add('Hill_coef',value=1.2,max=max_Hill_coef)
        
        asc_sigmoid_to_fit = sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
        
        
    
        first_Hill_result = first_Hill_model.fit(asc_sigmoid_to_fit, first_Hill_pars, x=original_x_data)
        
        
        best_H_Half_cst=first_Hill_result.best_values['Half_cst']
        best_H_Hill_coef=first_Hill_result.best_values['Hill_coef']
        best_H_x0 = first_Hill_result.best_values['x0']
        
        first_Hill_extended_fit = hill_function_new(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        first_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        first_Hill_extended_fit_table['Legend'] = "First_Hill_Fit"
        
        if do_plot:
            asc_sigmoid_fit_table = pd.DataFrame({'Stim_amp_pA':original_x_data,
                                                        'Frequency_Hz' : asc_sigmoid_to_fit})
            asc_sigmoid_fit_table['Legend'] = 'Ascending_Sigmoid'
            
            first_hill_fit_plot = ggplot()
            Hill_asc_table=first_Hill_extended_fit_table.append(asc_sigmoid_fit_table,ignore_index=True)
            first_hill_fit_plot += geom_line(Hill_asc_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            
            
            first_hill_fit_plot += ggtitle("Hill_Fit_to_ascending_Sigmoid")
            first_hill_fit_plot += scale_color_manual(values=scale_dict)
            
            print(first_hill_fit_plot)
            
        best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
        best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']       
        
    
        Hill_model = Model(hill_function_new, prefix='Hill_')
        Hill_pars = Parameters()
        Hill_pars.add("Hill_x0",value=best_H_x0,min=best_H_x0-0.5*best_H_Half_cst, max=best_H_x0+0.5*best_H_Half_cst)
        Hill_pars.add("Hill_Half_cst",value=best_H_Half_cst, min=0, max= 5*best_H_Half_cst)
        Hill_pars.add('Hill_Hill_coef',value=best_H_Hill_coef,vary = False)
        
        
        
        Sigmoid_model = Model(sigmoid_function, prefix="Sigmoid_")
        Sigmoid_pars = Parameters()
        
        Sigmoid_pars.add("Sigmoid_x0",value=best_value_x0_desc)
        Sigmoid_pars.add("Sigmoid_sigma",value=best_value_sigma_desc)
        
        Hill_Sigmoid_model = Hill_model*Sigmoid_model
        Hill_Sigmoid_pars = Hill_pars+Sigmoid_pars
        
        asc_Hill_desc_sigmoid_to_be_fit = hill_function_new(original_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        asc_Hill_desc_sigmoid_to_be_fit *= sigmoid_function(original_x_data , best_value_x0_desc, best_value_sigma_desc)
        
        # fit a Hill*sigmoid to First_Hill*Desc_Sigmoid
        
        Hill_Sigmoid_result = Hill_Sigmoid_model.fit(asc_Hill_desc_sigmoid_to_be_fit, Hill_Sigmoid_pars, x=original_x_data)
        
        H_Half_cst=Hill_Sigmoid_result.best_values['Hill_Half_cst']
        H_Hill_coef=Hill_Sigmoid_result.best_values['Hill_Hill_coef']
        H_x0 = Hill_Sigmoid_result.best_values['Hill_x0']
        
    
        S_x0 = Hill_Sigmoid_result.best_values['Sigmoid_x0']
        S_sigma = Hill_Sigmoid_result.best_values['Sigmoid_sigma']
        
        Hill_Sigmoid_model_results = hill_function_new(extended_x_data, H_x0, H_Hill_coef, H_Half_cst)
        Hill_Sigmoid_model_results *= sigmoid_function(extended_x_data , S_x0, S_sigma)
        
        
        
        asc_Hill_desc_sigmoid_extended = hill_function_new(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        asc_Hill_desc_sigmoid_extended *= sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
        asc_Hill_desc_sigmoid_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                  "Frequency_Hz" : asc_Hill_desc_sigmoid_extended})
        asc_Hill_desc_sigmoid_table['Legend'] = "Asc_Hill_Desc_Sigmoid"
        
        Hill_Sigmoid_model_results_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                  "Frequency_Hz" : Hill_Sigmoid_model_results})
        Hill_Sigmoid_model_results_table['Legend'] = 'Hill_Sigmoid_Fit'
        
        
        Hill_Sigmoid_plot =  ggplot()
        Hill_Sigmoid_plot += geom_line(asc_Hill_desc_sigmoid_table, aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += geom_line(Hill_Sigmoid_model_results_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += ggtitle("Hill_Sigmoid_Fit_to_Asc_Hill_Desc_Sigm")
        Hill_Sigmoid_plot += scale_color_manual(values=scale_dict)
        if do_plot:
            print(Hill_Sigmoid_plot)
        
        asc_Hill_desc_sigmoid_table_with_amp = asc_Hill_desc_sigmoid_table.copy()
        asc_Hill_desc_sigmoid_table_with_amp['Frequency_Hz'] *= best_value_Amp
        
        Hill_Sigmoid_model_results_table_with_amplitude = Hill_Sigmoid_model_results_table.copy()
        Hill_Sigmoid_model_results_table_with_amplitude['Frequency_Hz']*=best_value_Amp
        
        Hill_Sigmoid_plot =  ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
        #Hill_Sigmoid_plot += geom_line(asc_Hill_desc_sigmoid_table_with_amp, aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += geom_line(Hill_Sigmoid_model_results_table_with_amplitude,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += ggtitle("Final_Hill_Sigmoid_Fit_mytest")
        Hill_Sigmoid_plot += scale_color_manual(values=scale_dict)
        Hill_Sigmoid_plot += scale_shape_manual(values=scale_dict)
        
        if do_plot:
            print(Hill_Sigmoid_plot)
            
        
        

        Legend="Hill_Sigmoid_Fit"
        
        
        obs = 'Hill-Sigmoid'
        
        return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend
        
    except(StopIteration):
         obs='Error_Iteration'
    
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend
             
    except (ValueError):
          obs='Error_Value'
          best_value_Amp=np.nan
          best_H_Hill_coef=np.nan
          best_H_Half_cst=np.nan
          best_H_x0=np.nan
          best_value_x0_desc=np.nan
          best_value_sigma_desc=np.nan
          
          return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend
              
              
    except (RuntimeError):
         obs='Error_Runtime'
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend
             
    except (TypeError) as e:
         obs=str(e)
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend
             
             
def fit_Single_Hill(original_data_init, scale_dict,do_plot=False):
     
    try:


        original_data=original_data_init.copy()
        first_stimulus_frequency_table=original_data.copy()

        #x_shift=abs(min(first_stimulus_frequency_table['Stim_amp_pA']))
        #first_stimulus_frequency_table.loc[:,'Stim_amp_pA']=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']+x_shift
        
        if 'Passed_QC' in first_stimulus_frequency_table.columns:
            first_stimulus_frequency_table=first_stimulus_frequency_table[first_stimulus_frequency_table['Passed_QC']==True]
        

        first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data['Stim_amp_pA'])
        original_y_data = np.array(original_data['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))

    
        ### 0 - Trim Data
        response_threshold = (np.nanmax(first_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(first_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
        
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']
        trimmed_y_data=first_stimulus_frequency_table.loc[:,"Frequency_Hz"]
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        
        second_order_poly_model = PolynomialModel(degree = 2)
        pars = second_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        second_order_poly_model_results = second_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
        best_c0 = second_order_poly_model_results.best_values['c0']
        best_c1 = second_order_poly_model_results.best_values['c1']
        best_c2 = second_order_poly_model_results.best_values['c2']
        
        extended_trimmed_2nd_poly_model = best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        extended_trimmed_2nd_poly_model_freq_diff=np.diff(extended_trimmed_2nd_poly_model)
        trimmed_2nd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_2nd_poly_model})
        
       
        zero_crossings_2nd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_2nd_poly_model_freq_diff)))[0] # detect last index before change of sign
        
        if len(zero_crossings_2nd_poly_model_freq_diff) == 1:
    
            if extended_trimmed_2nd_poly_model_freq_diff[0]<0:
                trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] >= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
            else:
                trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] <= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
            ascending_segment = trimmed_2nd_poly_table.copy()
        else:    
            ascending_segment = trimmed_2nd_poly_table.copy()
        ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
        ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
        
        trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
        max_freq_poly_fit=np.nanmax(trimmed_2nd_poly_table['Frequency_Hz'])
        
        trimmed_2nd_poly_table.loc[trimmed_2nd_poly_table.index,'Legend']='2nd_order_poly'
        trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
        if do_plot:
            
            polynomial_plot = ggplot(first_stimulus_frequency_table,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
            polynomial_plot += geom_line(trimmed_2nd_poly_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += ggtitle('2nd order polynomial fit to trimmed_data')
            polynomial_plot += scale_shape_manual(values=scale_dict)
            print(polynomial_plot)
        ### 2 - end 
        
        ### 3 - Linear fit on polynomial trimmed data
        ascending_linear_slope_init,ascending_linear_intercept_init=fit_specimen_fi_slope(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        if do_plot:   
            polynomial_plot+=geom_abline(slope=ascending_linear_slope_init,
                                         intercept=ascending_linear_intercept_init,
                                         colour="pink",
                                         linetype='dashed')
            
        ### 3 - end
        
        ### 4 - Fit single or double Sigmoid
        
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        
        
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        
        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data , best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        if do_plot:
            
            
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data['Frequency_Hz'])/max(asc_sig_comp_table['Frequency_Hz'])
            amp_comp = double_sigmoid_comps['Amp_']
            amp_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : amp_comp,
                                        'Legend' : "Amplitude"})
            
            component_table = asc_sig_comp_table.append(amp_comp_table,ignore_index=True)
            component_plot = ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
           
            #trimmed_ascending_segment["Stim_amp_pA"]-=x_shift
            sigmoid_fit_plot =  ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
            sigmoid_fit_plot += geom_line(full_sigmoid_fit_table, aes(x='Stim_amp_pA',y="Frequency_Hz"))
            sigmoid_fit_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += ggtitle("Sigmoid_fit_to original_data")

            component_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            component_plot += ggtitle('Sigmoid_fit_components')
            component_plot += scale_color_manual(values=scale_dict)
            component_plot += scale_shape_manual(values=scale_dict)
            print(component_plot)
            
            
            sigmoid_fit_plot += scale_color_manual(values=scale_dict)
            sigmoid_fit_plot += scale_shape_manual(values=scale_dict)
            print(sigmoid_fit_plot)
        
        ### 4 - end
        
        ### 5 - Fit Hill function to ascending Sigmoid (without considering amplitude)
        
        #Determine Hill half coef for first hill fit --> First stim_amp to elicit half max(frequency)
        half_response_index=np.argmax(original_y_data>=(0.5*max(original_y_data)))
       
        half_response_stim = original_x_data[half_response_index]
        
        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
    
       

        first_Hill_model = Model(hill_function_new)
        first_Hill_pars = first_Hill_model.make_params()
        first_Hill_pars.add("x0",value=np.nanmin(original_x_data),min=-100)
        first_Hill_pars.add("Half_cst",value=half_response_stim,min=1e-9)
        first_Hill_pars.add('Hill_coef',value=1.2,max=max_Hill_coef)
        
        asc_sigmoid_to_fit = sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
        
        
    
        first_Hill_result = first_Hill_model.fit(asc_sigmoid_to_fit, first_Hill_pars, x=original_x_data)
        
        
        best_H_Half_cst=first_Hill_result.best_values['Half_cst']
        best_H_Hill_coef=first_Hill_result.best_values['Hill_coef']
        best_H_x0 = first_Hill_result.best_values['x0']
        
        first_Hill_extended_fit = hill_function_new(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        first_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        first_Hill_extended_fit_table['Legend'] = "First_Hill_Fit"
        
        if do_plot:
            asc_sigmoid_fit_table = pd.DataFrame({'Stim_amp_pA':original_x_data,
                                                        'Frequency_Hz' : asc_sigmoid_to_fit})
            asc_sigmoid_fit_table['Legend'] = 'Ascending_Sigmoid'
            
            first_hill_fit_plot = ggplot()
            Hill_asc_table=first_Hill_extended_fit_table.append(asc_sigmoid_fit_table,ignore_index=True)
            first_hill_fit_plot += geom_line(Hill_asc_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            
            
            first_hill_fit_plot += ggtitle("Hill_Fit_to_ascending_Sigmoid")
            first_hill_fit_plot += scale_color_manual(values=scale_dict)
            
            print(first_hill_fit_plot)
            
        Final_Hill_fit = hill_function_new(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        Final_Hill_fit=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        Final_Hill_fit['Legend'] = "Hill_Fit"
        Final_Hill_fit['Frequency_Hz'] *= best_value_Amp
        
        Final_Hill_plot =  ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point(aes(shape='Passed_QC'))
        
        Final_Hill_plot += geom_line(Final_Hill_fit,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Final_Hill_plot += ggtitle("Final_Hill_Fit")
        Final_Hill_plot += scale_color_manual(values=scale_dict)
        Final_Hill_plot += scale_shape_manual(values=scale_dict)
        if do_plot:
            print(Final_Hill_plot)
            
        
        
        Legend='Hill_Fit'

        obs = 'Hill'
        
        
        return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend
        
    except(StopIteration):
         obs='Error_Iteration'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'
         
         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend
             
    except (ValueError):
          obs='Error_Value'
          Final_Hill_fit=np.nan
          best_value_Amp=np.nan
          best_H_Hill_coef=np.nan
          best_H_Half_cst=np.nan
          best_H_x0=np.nan
          Legend='--'
          
          
          return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend
              
    except (RuntimeError):
         obs='Error_Runtime'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'
         
         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend
             
    except (TypeError):
         obs='Type_Error'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'

         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend
             
def extract_IO_features(original_data,x_shift,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,do_plot=False):
    
    try:
        scale_dict={True:"o",
                    False:'s'}
        if Descending_segment:
            obs='Hill-Sigmoid'
        else:
            obs='Hill'
        
        
        original_data_shifted=original_data.copy()
        #original_data_shifted.loc[:,'Stim_amp_pA']=original_data_shifted.loc[:,'Stim_amp_pA']+x_shift
        
        if 'Passed_QC' in original_data_shifted.columns:
            original_data_shifted=original_data_shifted[original_data_shifted['Passed_QC']==True]
        
        
        
        original_x_data = np.array(original_data_shifted['Stim_amp_pA'])
        original_y_data = np.array(original_data_shifted['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
        predicted_y_data = fit_model_table['Frequency_Hz']

        my_derivative = np.array(predicted_y_data)[1:]-np.array(predicted_y_data)[:-1]
    
    
        #twentyfive_index=next(x for x, val in enumerate(predicted_y_data) if val >(0.25*max(predicted_y_data)))
        
        twentyfive_index=np.argmax(predicted_y_data>(0.25*max(predicted_y_data)))
    
        seventyfive_index=np.argmax(predicted_y_data>(0.75*max(predicted_y_data)))                         
        #seventyfive_index=next(x for x, val in enumerate(predicted_y_data) if val >(0.75*max(predicted_y_data)))
    
    
        Gain,Intercept_shift=fit_specimen_fi_slope(extended_x_data.iloc[twentyfive_index:seventyfive_index],predicted_y_data.iloc[twentyfive_index:seventyfive_index])


        Threshold_shift=(0-Intercept_shift)/Gain
        #Intercept=Gain*x_shift+Intercept_shift
        Intercept=Gain+Intercept_shift

        #Threshold=Threshold_shift-x_shift
        Threshold=Threshold_shift
        #extended_x_data-=x_shift
        model_table=pd.DataFrame(np.column_stack((extended_x_data,predicted_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
        model_table['Legend']=Legend
        
        if np.nanmean(my_derivative[-10:]) <= np.nanmax(my_derivative[twentyfive_index:seventyfive_index])*.01:
            Saturation_frequency = np.nanmax(predicted_y_data)
    
            sat_model_table=model_table[model_table["Frequency_Hz"] == Saturation_frequency]
            Saturation_frequency=sat_model_table['Frequency_Hz'].values[0]
            Saturation_stimulation=sat_model_table['Stim_amp_pA'].values[0]
            
        else:
            
            Saturation_frequency=np.nan
            Saturation_stimulation=np.nan
            
            
        ### Compute QNRMSE
        
        if len(np.flatnonzero(original_y_data))>0:
            without_zero_index=np.flatnonzero(original_y_data)[0]
            last_zero_index = without_zero_index-1
        else:
            without_zero_index=original_y_data.iloc[0]

        sub_x_data=original_x_data[without_zero_index:]
        sub_y_data=original_y_data[without_zero_index:]
        new_x_data_without_zero=pd.Series(np.arange(min(sub_x_data),max(sub_x_data),1))
        
        pred = hill_function_new(sub_x_data, Hill_x0, Hill_coef, Hill_Half_cst)
        
        
        pred *= Amp
        
        pred_without_zero = hill_function_new(new_x_data_without_zero, Hill_x0, Hill_coef, Hill_Half_cst)
        
        pred_without_zero *= Amp
        
        if Descending_segment:
            pred *= sigmoid_function(sub_x_data , sigmoid_x0, sigmoid_sigma)
            pred_without_zero *= sigmoid_function(new_x_data_without_zero ,sigmoid_x0, sigmoid_sigma)
        
        best_QNRMSE=normalized_root_mean_squared_error(sub_y_data,pred,pred_without_zero)

        if do_plot:
            #original_data_shifted['Stim_amp_pA']-=x_shift
            feature_plot=ggplot(original_data,aes(x='Stim_amp_pA',y='Frequency_Hz'))+geom_point(aes(shape='Passed_QC'))
            
            if best_QNRMSE >= .5:
                feature_plot+=geom_line(model_table,aes(x='Stim_amp_pA',y='Frequency_Hz',color='Legend',group='Legend'),linetype='dashed')
            else:
                feature_plot+=geom_line(model_table,aes(x='Stim_amp_pA',y='Frequency_Hz',color='Legend',group='Legend'))
                feature_plot+=geom_abline(aes(intercept=Intercept,slope=Gain))
                Threshold_table=pd.DataFrame({'Stim_amp_pA':[Threshold],'Frequency_Hz':[0]})
                feature_plot+=geom_point(Threshold_table,aes(x=Threshold_table["Stim_amp_pA"],y=Threshold_table["Frequency_Hz"]),color='green')
    
                if not np.isnan(Saturation_frequency):
                    
                    saturation_table=pd.DataFrame(columns=["x1","x2","Sat_stim","y1","y2","Sat_Freq"])
                    saturation_series=pd.Series([Saturation_stimulation-20,Saturation_stimulation+20,Saturation_stimulation,
                                                 Saturation_frequency-5,Saturation_frequency+5,Saturation_frequency],
                                                index=["x1","x2","Sat_stim","y1","y2","Sat_Freq"])
                    saturation_table=saturation_table.append(saturation_series,ignore_index=True)
    
                    
                    feature_plot += geom_segment(saturation_table,aes(x="x1", xend="x2",y="Sat_Freq", yend="Sat_Freq"))
                    feature_plot += geom_segment(saturation_table,aes(x="Sat_stim", xend="Sat_stim",y="y1", yend="y2"))
            feature_plot += scale_shape_manual(values=scale_dict)
            
            print(feature_plot)
    
        if best_QNRMSE >=.5:
            obs='QNRMSE_too_high'
            Gain=np.nan
            Threshold=np.nan
            Saturation_frequency=np.nan
            Saturation_stimulation=np.nan
            
        return obs,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation
            
    except (TypeError) as e:
        
        obs=str(e)
        
        best_QNRMSE=np.nan
        x_shift=np.nan
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        
        return obs,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation
    except (ValueError) as e:
        
        obs=str(e)
        
        best_QNRMSE=np.nan
        x_shift=np.nan
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        return obs,best_QNRMSE,x_shift,Gain,Threshold,Saturation_frequency,Saturation_stimulation
    
def extract_inst_freq_table(original_SF_table, original_cell_sweep_info_table,sweep_QC_table_inst,response_time):
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
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes

            
            
    new_columns=["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]

    
    #SF_table[new_columns] = np.nan
    SF_table = SF_table.reindex(SF_table.columns.tolist() + new_columns ,axis=1)
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

                new_line=pd.DataFrame([int(interval)+1, # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line]).T# Sweep id
                                   
                new_line.columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep']
                interval_freq_table=pd.concat([interval_freq_table,new_line],ignore_index=True)
                
    
    interval_freq_table = pd.merge(interval_freq_table,sweep_QC_table,on='Sweep')
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
   
    
    return  A*np.exp(-(x)/B)+C


def fit_adaptation_curve(interval_frequency_table_init,do_plot=False):
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
         interval_frequency_table = interval_frequency_table_init.copy()
         interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
         if interval_frequency_table.shape[0]==0:
             obs='Not_enough_spike'

             best_A=np.nan
             best_B=np.nan
             best_C=np.nan
             RMSE=np.nan
             return obs,best_A,best_B,best_C,RMSE
         interval_frequency_table.loc[:,'Spike_Interval']=interval_frequency_table.loc[:,'Spike_Interval'].astype(float)
         interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                                   "Normalized_Inst_frequency":"float",
                                                                   "Stimulus_amp_pA":'float'})
         x_data=interval_frequency_table.loc[:,'Spike_Interval']
         x_data=x_data.astype(float)
         y_data=interval_frequency_table.loc[:,'Normalized_Inst_frequency']
         y_data=y_data.astype(float)

         
         median_table=interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).median()
         median_table["Count_weights"]=pd.DataFrame(interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
         median_table["Spike_Interval"]=median_table.index
         median_table["Spike_Interval"]=np.float64(median_table["Spike_Interval"])  
         
         y_delta=y_data.iloc[-1]-y_data.iloc[0]
        
         y_delta_two_third=y_data.iloc[0]-.66*y_delta
         
         initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
         initial_time_cst_guess=x_data[initial_time_cst_guess_idx]
         
         
         
         
         
         initial_A=(y_data.iloc[0]-y_data.iloc[-1])/np.exp(-x_data.iloc[0]/initial_time_cst_guess)

         decayModel=Model(exponential_decay_function)

         decay_parameters=Parameters()

         decay_parameters.add("A",value=initial_A)
         decay_parameters.add('B',value=initial_time_cst_guess,min=0)
         decay_parameters.add('C',value=median_table["Normalized_Inst_frequency"][max(median_table["Spike_Interval"])])

         result = decayModel.fit(y_data, decay_parameters, x=x_data)

         best_A=result.best_values['A']
         best_B=result.best_values['B']
         best_C=result.best_values['C']


         pred=exponential_decay_function(np.array(x_data),best_A,best_B,best_C)
         squared_error = np.square((y_data - pred))
         sum_squared_error = np.sum(squared_error)
         RMSE = np.sqrt(sum_squared_error / y_data.size)

         A_norm=best_A/(best_A+best_C)
         C_norm=best_C/(best_A+best_C)
         interval_range=np.arange(1,max(median_table["Spike_Interval"])+1,.1)

         simulation=exponential_decay_function(interval_range,best_A,best_B,best_C)
         norm_simulation=exponential_decay_function(interval_range,A_norm,best_B,C_norm)
         sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])
         norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])



         my_plot=np.nan
         if do_plot==True:

             my_plot=ggplot(interval_frequency_table,aes(x=interval_frequency_table["Spike_Interval"],y=interval_frequency_table["Normalized_Inst_frequency"]))+geom_point(aes(color=interval_frequency_table["Stimulus_amp_pA"]))

             my_plot=my_plot+geom_point(median_table,aes(x='Spike_Interval',y='Normalized_Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
             my_plot=my_plot+geom_line(sim_table,aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color='black')
             my_plot=my_plot+geom_line(norm_sim_table,aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color="green")

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
     # except(TypeError):
     #     obs='Error_Type'
     #     A_norm=np.nan
     #     best_B=np.nan
     #     C_norm=np.nan
     #     RMSE=np.nan
     #     return obs,A_norm,best_B,C_norm,RMSE
     
def fit_adaptation_curve_old(original_interval_frequency_table,do_plot=False):
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

def fit_membrane_trace (original_time_membrane_table,start_time,end_time,stim_start_time,stim_end_time,do_plot=False):
    try:
        time_membrane_table=original_time_membrane_table.copy()
        x_data=np.array(time_membrane_table.loc[:,'Time_s'])
        y_data=np.array(time_membrane_table.loc[:,"Membrane_potential_mV"])
        start_idx = np.argmin(abs(x_data - start_time))
        end_idx = np.argmin(abs(x_data - end_time))
        
        
        
        
        y_delta=y_data[-1]-y_data[0]
        y_delta_two_third=y_data[0]+.66*y_delta
            
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        initial_time_cst_guess=(x_data[initial_time_cst_guess_idx]-x_data[0])
        
        initial_A=(y_data[0]-y_data[-1])/np.exp(-x_data[0]/initial_time_cst_guess)
        
       
    
        time_cst_Model=Model(time_cst_model)
    
        time_cst_parameters=Parameters()

        time_cst_parameters.add("A",value=initial_A)
        time_cst_parameters.add('tau',value=initial_time_cst_guess,min=0)
        time_cst_parameters.add('C',value=y_data[-1])

        result = time_cst_Model.fit(y_data[start_idx:end_idx], time_cst_parameters, x=x_data[start_idx:end_idx])
        best_A=result.best_values['A']
        best_tau=result.best_values['tau']
        best_C=result.best_values['C']
        #print(result.best_values)
        
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
    




def Heaviside_cst_exp_function(x, stim_start, stim_end,baseline,A,tau,C):
    """Heaviside step function."""
    
    if stim_end<=min(x):
        o=np.empty(x.size);o.fill(C)
        return o
    
    elif stim_start>=max(x):
        o=np.empty(x.size);o.fill(baseline)
        return o
    
    else:
        o=np.empty(x.size);o.fill(baseline)
        
        
        start_index = max(np.where( x < stim_start)[0])

        end_index=max(np.where( x < stim_end)[0])
        o[:start_index]=baseline
        o[start_index:end_index] = time_cst_model(x[start_index:end_index],A,tau,C)
        
        #o[end_index:] = time_cst_model(x[end_index:],A_bis,tau,C_bis)
    
        return o
    
def fit_membrane_time_cst (original_TPC_table,start_time,end_time,do_plot=False):
    try:
        TPC_table=original_TPC_table.copy()
        sub_TPC_table=TPC_table[TPC_table['Time_s']<=(end_time+.5)]
        sub_TPC_table=sub_TPC_table[sub_TPC_table['Time_s']>=(start_time-.5)]
        
        x_data=np.array(sub_TPC_table.loc[:,'Time_s'])
        y_data=np.array(sub_TPC_table.loc[:,"Membrane_potential_mV"])
        
        start_idx = np.argmin(abs(x_data - start_time))
        end_idx = np.argmin(abs(x_data - end_time))
        
        
        #Estimate parameters initial values
        membrane_resting_potential=np.median(y_data[:start_idx])

        mid_idx=int((end_idx+start_idx)/2)

        
        x_0=start_time
        y_0=y_data[np.argmin(abs(x_data - x_0))]
        
        x_1=start_time+.05
        y_1=y_data[np.argmin(abs(x_data - x_1))]
        
       
        initial_membrane_SS=np.median(y_data[mid_idx:end_idx])


        if (y_1-initial_membrane_SS)/(y_0-initial_membrane_SS)>0:
            initial_time_cst = (x_0-x_1)/np.log((y_1-initial_membrane_SS)/(y_0-initial_membrane_SS))
            initial_A=(y_1-initial_membrane_SS)*np.exp(x_1/initial_time_cst)


        else:

            membrane_delta=initial_membrane_SS-membrane_resting_potential
        
            
            initial_potential_time_cst=membrane_resting_potential+(2/3)*membrane_delta

            initial_potential_time_cst_idx=np.argmin(abs(y_data[start_idx:end_idx] - initial_potential_time_cst))+start_idx
            initial_time_cst=x_data[initial_potential_time_cst_idx]-start_time
            
        
            
            initial_A=(y_data[start_idx]-initial_membrane_SS)/np.exp(-x_data[start_idx]/initial_time_cst)
            
            
            
        double_step_model=Model(time_cst_model)
       
        double_step_model_pars=Parameters()
        
        double_step_model_pars.add('A',value=initial_A)
        double_step_model_pars.add('tau',value=initial_time_cst)
        double_step_model_pars.add('C',value=initial_membrane_SS)


        double_step_out=double_step_model.fit(y_data[start_idx:end_idx], double_step_model_pars, x=x_data[start_idx:end_idx])        

        best_A=double_step_out.best_values['A']
        best_tau=double_step_out.best_values['tau']
        best_C=double_step_out.best_values['C']
        
        
        simulation=time_cst_model(x_data[start_idx:end_idx],best_A,best_tau,best_C)
        sim_table=pd.DataFrame(np.column_stack((x_data[start_idx:end_idx],simulation)),columns=["Time_s","Membrane_potential_mV"])
        
        squared_error = np.square(y_data[start_idx:end_idx] - simulation)
        sum_squared_error = np.sum(squared_error)
        current_RMSE = np.sqrt(sum_squared_error / len(y_data[start_idx:end_idx]))
        
        NRMSE=current_RMSE/abs(membrane_resting_potential-initial_membrane_SS)


        
        
        if do_plot==True:
            
            my_plot=ggplot(TPC_table,aes(x=TPC_table["Time_s"],y=TPC_table["Membrane_potential_mV"]))+geom_line(color='blue')#+xlim((start_time-.1),(end_time+.1))
            
            
            my_plot=my_plot+geom_line(sim_table,aes(x='Time_s',y='Membrane_potential_mV'),color='red')
            
    
            print(my_plot)
            
        return best_A,best_tau,best_C,membrane_resting_potential,NRMSE
    except (TypeError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        NRMSE=np.nan
        return best_A,best_tau,best_C,membrane_resting_potential,NRMSE
    except(ValueError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        NRMSE=np.nan
        return best_A,best_tau,best_C,membrane_resting_potential,NRMSE
    

def fit_second_order_poly(original_fit_table,do_plot=False):
    try:
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
            
            my_plot=ggplot(fit_table,aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+geom_line()
            my_plot+=geom_line(simulation_table,aes(x="Time_s",y="Membrane_potential_mV",color='Data'))
            print(my_plot)
        return a,b,c,RMSE_poly
    
    except (ValueError):
        a = np.nan
        b = np.nan
        c = np.nan
        RMSE_poly = np.nan
        return  a,b,c,RMSE_poly

def fit_exponential_BE(original_fit_table,stim_start=True,do_plot=False):
    try:
        fit_table=original_fit_table.copy()
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        expo_model=ExponentialModel()
        
        
        membrane_start_potential = y_data[0]
        membrane_end_potential = y_data[-1]
        
        membrane_delta=membrane_end_potential - membrane_start_potential
    
        
        if stim_start == True:
            
            exp_offset = y_data[-1]
            initial_potential_time_cst=membrane_end_potential+(1-(1/np.exp(1)))*membrane_delta
            initial_potential_time_cst_idx=np.argmin(abs(y_data - initial_potential_time_cst))
            initial_time_cst=x_data[initial_potential_time_cst_idx]-x_data[0]
            initial_A=(membrane_start_potential-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            

                
        elif stim_start == False:

            
            
            exp_offset = y_data[-1]
            initial_potential_time_cst=membrane_end_potential+(1-(1/np.exp(1)))*membrane_delta
            initial_potential_time_cst_idx=np.argmin(abs(y_data - initial_potential_time_cst))
            initial_time_cst=x_data[initial_potential_time_cst_idx]-x_data[0]
            
            initial_time_cst= -initial_time_cst
            initial_A=(membrane_start_potential-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            
        
        
    
    
        expo_model = Model(time_cst_model)
        expo_model_pars=Parameters()
        expo_model_pars.add('A',value=initial_A)
        expo_model_pars.add('tau',value=initial_time_cst)
        expo_model_pars.add('C',value=exp_offset)
        
        expo_out=expo_model.fit(y_data, expo_model_pars, x=x_data)        
        
        
        best_A=expo_out.best_values['A']
        best_tau=expo_out.best_values['tau']
        best_C=expo_out.best_values['C']
        
        
        
        pred = time_cst_model(x_data,best_A,best_tau,best_C)
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
        
        return best_A,best_tau,RMSE_expo
    
    except (ValueError):
        best_A=np.nan
        best_tau=np.nan
        RMSE_expo=np.nan
        return best_A,best_tau,RMSE_expo
    

def Double_Heaviside_function(x, stim_start, stim_end,baseline,stim_amplitude):
    """Heaviside step function."""
    
    if stim_end<=min(x):
        o=np.empty(x.size);o.fill(stim_amplitude)
        return o
    
    elif stim_start>=max(x):
        o=np.empty(x.size);o.fill(baseline)
        return o
    
    else:
        o=np.empty(x.size);o.fill(baseline)
        
        
        start_index = max(np.where( x < stim_start)[0])

        end_index=max(np.where( x < stim_end)[0])
        
        o[start_index:end_index] = stim_amplitude
    
        return o

def fit_stimulus_trace(TPC_table_original,stim_start,stim_end,do_plot=False):
    
    TPC_table=TPC_table_original.copy()
    stim_table=TPC_table.loc[:,['Time_s','Input_current_pA']].copy()
    stim_table = stim_table.reset_index(drop=True)
    x_data=stim_table.loc[:,"Time_s"]
    index_stim_start=next(x for x, val in enumerate(x_data[0:]) if val >= stim_start )
    index_stim_end=next(x for x, val in enumerate(x_data[0:]) if val >= (stim_end) )
    
    
    
    stim_table['Input_current_pA']=np.array(data_treat.filter_trace(stim_table['Input_current_pA'],
                                                                        stim_table['Time_s'],
                                                                        filter=1.,
                                                                        do_plot=False))
    
    first_current_derivative=data_treat.get_derivative(np.array(stim_table["Input_current_pA"]),np.array(stim_table["Time_s"]))
    first_current_derivative=np.insert(first_current_derivative,0,first_current_derivative[0])
    
    stim_table['Filtered_Stimulus_trace_derivative_pA/ms']=np.array(first_current_derivative)
    
    
    
    before_stim_start_table=stim_table[stim_table['Time_s']<stim_start-.01]
    baseline_current=np.mean(before_stim_start_table['Input_current_pA'])
    
    during_stimulus_table=stim_table[stim_table['Time_s']<stim_end]
    during_stimulus_table=during_stimulus_table[during_stimulus_table['Time_s']>stim_start]
    estimate_stim_amp=np.median(during_stimulus_table.loc[:,'Input_current_pA'])
    
    
    
    y_data=stim_table.loc[:,"Input_current_pA"]
    

    stim_table['weight']=np.abs(1/stim_table['Filtered_Stimulus_trace_derivative_pA/ms'])**2
    
    #stim_table.replace([np.inf, -np.inf], np.nanmin(stim_table['weight']), inplace=True)
    
    
    weight=stim_table.loc[:,"weight"]

    if np.isinf(weight).sum!=0:
        
        if np.isinf(weight).sum() >= (len(weight)/2): # if inf values represent more than half the values
            weight=np.ones(len(weight))

        else: # otherwise replace inf values by the maximum value non inf

            max_weight_without_inf=np.nanmax(weight[weight != np.inf])
            weight.replace([np.inf], max_weight_without_inf, inplace=True)
   
    weight/=np.nanmax(weight) # normalize the weigth to the maximum weight 

    double_step_model=Model(Double_Heaviside_function)
    double_step_model_parameters=Parameters()
    double_step_model_parameters.add('stim_start',value=stim_start,vary=False)
    double_step_model_parameters.add('stim_end',value=stim_end,vary=False)
    double_step_model_parameters.add('baseline',value=baseline_current)
    double_step_model_parameters.add('stim_amplitude',value=estimate_stim_amp)


    #return stim_table
    double_step_out=double_step_model.fit(y_data,double_step_model_parameters,x=x_data,weights=weight)
    

    
    best_baseline=double_step_out.best_values['baseline']
    best_stim_amp=double_step_out.best_values['stim_amplitude']
    best_stim_start=double_step_out.best_values['stim_start']
    best_stim_end=double_step_out.best_values['stim_end']
    
    NRMSE_double_Heaviside=mean_squared_error(y_data.iloc[index_stim_start:index_stim_end], Double_Heaviside_function(x_data, best_stim_start, best_stim_end, best_baseline, best_stim_amp)[index_stim_start:index_stim_end],squared=(False))/(best_stim_amp)

    if do_plot:
        computed_y_data=pd.Series(Double_Heaviside_function(x_data, best_stim_start,best_stim_end, best_baseline, best_stim_amp))
        model_table=pd.DataFrame({'Time_s' :x_data,
                                  'Input_current_pA': computed_y_data})#np.column_stack((x_data,computed_y_data)),columns=['Time_s ',"Stim_amp_pA"])
        
        original_table=pd.DataFrame(np.column_stack((x_data,y_data)),columns=['Time_s ',"Stim_amp_pA"])
        TPC_table['Legend']='Original_Data'
        
        stim_table['Legend']='Filtered_fitted_Data'
        model_table['Legend']='Fit'
        color_dict = {'Original_Data' : 'black',
                      'Filtered_fitted_Data' : 'pink',
                      'Fit' : 'blue'}
        data_table = TPC_table.append(stim_table,ignore_index=True)
        data_table = data_table.append(model_table, ignore_index=True)
        # my_plot = ggplot()+geom_line(data_table,aes(x='Time_s', y= 'Input_current_pA',group='Legend',color='Legend'))+scale_color_manual(color_dict)
        # my_plot+=xlim((stim_start-0.05), (stim_end+0.05))
        # my_plot+=ylim((best_baseline), (best_stim_amp+50))
        # print(my_plot)
        # my_plot=ggplot(original_table,aes(x=original_table.iloc[:,0],y=original_table.iloc[:,1]))+geom_line(color='red')
        my_plot = ggplot()
        my_plot+=geom_line(TPC_table,aes(x='Time_s',y='Input_current_pA'),colour = 'black')
        my_plot+=geom_line(stim_table,aes(x='Time_s',y='Input_current_pA'),colour = 'red')
        my_plot+=geom_line(model_table,aes(x='Time_s',y='Input_current_pA'),colour='blue')
        my_plot+=xlab(str("Time_s"))
        my_plot+=xlim((stim_start-0.05), (stim_end+0.05))
        
        print(my_plot)
    return best_baseline,best_stim_amp,best_stim_start,best_stim_end,NRMSE_double_Heaviside



