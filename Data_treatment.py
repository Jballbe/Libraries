#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:51:51 2022

@author: julienballbe
"""

from multiprocessing import Pool
import concurrent.futures
from datetime import date
import glob
import h5py
import functools
import Fit_library as fitlib
import Electrophy_treatment as ephys_treat
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
from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels, geom_histogram, ggtitle, geom_vline,geom_hline
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
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model, ExponentialModel, ConstantModel,PolynomialModel
from lmfit import Parameters, Minimizer, fit_report
from plotnine.scales import scale_y_continuous, ylim, xlim, scale_color_manual
from plotnine.labels import xlab
from plotnine.stats import stat_bin
from sklearn.metrics import mean_squared_error
import seaborn as sns
import sys
import math
import Parallel_functions as parafunc

sys.path.insert(
    0, '/Users/julienballbe/My_Work/My_Librairies/Electrophy_treatment.py')

# %%
ctc = CellTypesCache(
    manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
# %%


def create_cell_file(cell_id_list, saving_folder_path="/Volumes/Work_Julien/Cell_Data_File/", selection=['All'], overwrite=False):
    '''
    Generate or modify for a list of cell_id, corresponding HDF5 files.
    This function call the appropriate functions to create or re-create data according to needed features in selection
    For every processed cell, the date of last modification of cell item will be written in Full_database_table csv

    Parameters
    ----------
    cell_id_list : list
        List of cell_id for which we want to create or modify cell file.

    saving_folder_path : str, optional
        File path to which store cell file. The default is "/Volumes/Work_Julien/Cell_Data_File/".
    selection : list of str, optional
        List of item to modify or create for all cells present in cell_id_list. The default is ['All'].
    overwrite : Bool, optional
        If for a given cell_id, a cell file already exists in saving_folder_path, wether to overwrite already existing information.
        The default is False.

    Raises
    ------
    Exception
        If any problem occured during the process, the error is noted into Full_database_table.

    Returns
    -------
    str
        return Done when all cells have been treated.

    '''
    today = date.today()
    today = today.strftime("%Y_%m_%d")
    original_files = [f for f in glob.glob(str(saving_folder_path+'*.h5*'))]
    population_class_table = pd.read_csv(
        "/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Population_Class.csv")
    population_class_table = population_class_table.copy()
    Full_database_table = pd.read_csv(
        '/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Database_Cell_id.csv')
    Full_database_table = Full_database_table.copy()
    problem_cell=[]

    for cell_id in tqdm(cell_id_list):
        Cell_file_information = pd.read_csv(
            "/Volumes/Work_Julien/Cell_Data_File/Cell_file_information.csv",index_col=None)
        Cell_file_information = Cell_file_information.copy()

        cell_file_path = str(saving_folder_path+'Cell_' +
                             str(cell_id)+'_data_file.h5')
        if cell_id not in Cell_file_information['Cell_id']:
            empty_array = np.empty(Cell_file_information.shape[1]-1)
            empty_array[:] = np.nan
            new_cell_id_line = pd.DataFrame([cell_id,*empty_array]).T
            new_cell_id_line.columns=Cell_file_information.columns
            Cell_file_information = pd.concat([Cell_file_information,new_cell_id_line],ignore_index=True)
            
        cell_index = Cell_file_information[Cell_file_information['Cell_id'] == str(
            cell_id)].index.tolist()
        try:
            if cell_file_path in original_files and 'All' in selection and overwrite == True:  # rewrite the whole file

                cell_database = Full_database_table[Full_database_table['Cell_id'] == str(
                    cell_id)]['Database'].values[0]

                metadata_table, Full_TPC_table, Full_SF_table, Full_SF_dict, cell_sweep_info_table, cell_sweep_QC,cell_fit_table, cell_feature_table = process_cell_data(str(cell_id),
                                                                                                                                                                                      cell_database,
                                                                                                                                                                                      population_class_table,
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      selection=selection)

                os.remove(cell_file_path)
                write_cell_file_h5(cell_file_path,
                                   Full_TPC_table,
                                   Full_SF_dict,
                                   cell_sweep_info_table,
                                   cell_sweep_QC,
                                   cell_fit_table,
                                   cell_feature_table,
                                   metadata_table,
                                   selection=selection,
                                   overwrite=overwrite)

                Cell_file_information.loc[cell_index, 'Note'] = '--'
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_TPC'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_SF'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_Sweep_info'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_Sweep_QC'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_Metadata'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_Cell_Feature'] = str(today)
                Cell_file_information.loc[cell_index,
                                          'Lastly_modified_Cell_Fit'] = str(today)

            elif cell_file_path in original_files and 'All' not in selection and overwrite == True:  # rewrite only part of the file
                cell_database = Full_database_table[Full_database_table['Cell_id'] == str(
                    cell_id)]['Database'].values[0]
                import_dict={'TPC_SF':['TPC_SF','Sweep_info'],
                             'Sweep_info':['TPC_SF'],
                             'Sweep_QC':['Sweep_info'],
                             'Metadata':['Sweep_info'],
                             'Cell_Feature_Fit':['TPC_SF','Sweep_info']
                    }
                needed_list=[]

                for elt in selection:
                    needed_list=list(set(needed_list)|set(import_dict[elt]))
                #
                # needed_list = list(
                #     set(['Metadata', 'Sweep_info','Sweep_QC', "Cell_Feature_Fit"])-set(selection))
                # needed_list.append('TPC_SF')

                Full_TPC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table,cell_sweep_QC, cell_fit_table, cell_feature_table = import_h5_file(
                    cell_file_path, selection=needed_list)

                if Full_TPC_table.empty or Full_SF_dict_table.empty or Full_SF_table.empty:
                    raise Exception('TPC_table_or_SF_tables_missing')

                metadata_table, Full_TPC_table, Full_SF_table, Full_SF_dict, cell_sweep_info_table,cell_sweep_QC, cell_fit_table, cell_feature_table = process_cell_data(str(cell_id),
                                                                                                                                                                                      cell_database,
                                                                                                                                                                                      population_class_table,
                                                                                                                                                                                      Metadata_table,
                                                                                                                                                                                      Full_TPC_table,
                                                                                                                                                                                      Full_SF_table,
                                                                                                                                                                                      Full_SF_dict_table,
                                                                                                                                                                                      sweep_info_table,
                                                                                                                                                                                      cell_sweep_QC,
                                                                                                                                                                                      cell_fit_table,
                                                                                                                                                                                      cell_feature_table,
                                                                                                                                                                                      selection=selection)

                write_cell_file_h5(cell_file_path,
                                   Full_TPC_table,
                                   Full_SF_dict,
                                   cell_sweep_info_table,
                                   cell_sweep_QC,
                                   cell_fit_table,
                                   cell_feature_table,
                                   metadata_table,
                                   selection=selection,
                                   overwrite=overwrite)
                Cell_file_information.loc[cell_index, 'Note'] = '--'
                if 'TPC_SF' in selection:
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_TPC'] = str(today)
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_SF'] = str(today)
                    Cell_file_information.loc[cell_index, 'Lastly_modified_Sweep_info'] = str(
                        today)
                if 'Metadata' in selection:
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Metadata'] = str(today)
                if 'Cell_Feature_Fit' in selection:
                    Cell_file_information.loc[cell_index, 'Lastly_modified_Cell_Feature'] = str(
                        today)
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Cell_Fit'] = str(today)
                if 'Sweep_info' in selection:
                    Cell_file_information.loc[cell_index, 'Lastly_modified_Sweep_info'] = str(
                        today)
                if 'Sweep_QC' in selection:
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Sweep_QC'] = str(today)

            elif cell_file_path in original_files and overwrite == False:
                print(str('File for cell '+str(cell_id)+' already existing'))

            else:  # create file

                cell_database = Full_database_table[Full_database_table['Cell_id'] == str(
                    cell_id)]['Database'].values[0]

                metadata_table, Full_TPC_table, Full_SF_table, Full_SF_dict, cell_sweep_info_table,cell_sweep_QC, cell_fit_table, cell_feature_table = process_cell_data(str(cell_id),
                                                                                                                                                                                      cell_database,
                                                                                                                                                                                      population_class_table,
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      '-',
                                                                                                                                                                                      selection=selection)

                write_cell_file_h5(cell_file_path,
                                   Full_TPC_table,
                                   Full_SF_dict,
                                   cell_sweep_info_table,
                                   cell_sweep_QC,
                                   cell_fit_table,
                                   cell_feature_table,
                                   metadata_table,
                                   selection=selection,
                                   overwrite=overwrite)
                Cell_file_information.loc[cell_index, 'Note'] = '--'
                
                if 'TPC_SF' in selection:
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_TPC'] = str(today)
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_SF'] = str(today)
                    Cell_file_information.loc[cell_index, 'Lastly_modified_Sweep_info'] = str(
                        today)
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Sweep_QC'] = str(today)
                if 'Metadata' in selection:
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Metadata'] = str(today)
                if 'Cell_Feature_Fit' in selection:
                    Cell_file_information.loc[cell_index, 'Lastly_modified_Cell_Feature'] = str(
                        today)
                    Cell_file_information.loc[cell_index,
                                              'Lastly_modified_Cell_Fit'] = str(today)

        except Exception as e:
            problem_cell.append(str(cell_id))
            print(str('Problem_with_cell:'+str(cell_id)+str(e)))
            Cell_file_information.loc[cell_index, 'Note'] = str(type(e))

        Cell_file_information = Cell_file_information.loc[:, ['Cell_id',
                                                              'Database',
                                                              'Note',
                                                              'Lastly_modified_TPC',
                                                              'Lastly_modified_SF',
                                                              'Lastly_modified_Sweep_info',
                                                              "Lastly_modified_Sweep_QC",
                                                              'Lastly_modified_Metadata',
                                                              'Lastly_modified_Cell_Feature',
                                                              'Lastly_modified_Cell_Fit']]
        Cell_file_information.to_csv(
            "/Volumes/Work_Julien/Cell_Data_File/Cell_file_information.csv")
    print('Finished')
    return problem_cell


def process_cell_data(cell_id,
                      database,
                      population_class_file,
                      original_metadata_table='-',
                      original_Full_TPC_table = '-',
                      original_Full_SF_table = '-',
                      original_Full_SF_dict ='-',
                      original_cell_sweep_info_table = '-',
                      original_cell_sweep_QC_table = '-',
                      original_cell_fit_table = '-',
                      original_cell_feature_table = '-',
                      selection=['All']):
    '''
    This function  calls corresponding function to generate cell item present in selection.

    Parameters
    ----------
    cell_id : str
        Cell_id.
    database : str
        The database from which the cell_id is coming.
    population_class_file : pd.Dataframe
        Table gathering for Every cell different information used for Metadata table (Area, layer...) .
    original_metadata_table : pd.DataFrame

    original_Full_TPC_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('TPC') contains a TPC pd.DataFrame (Contains Time, Current,Potential, First Time deriv of Potential, and Second Time deriv of Potential)
    original_Full_SF_table : pd.DataFrame
        DESCRIPTION.
    original_Full_SF_dict : DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TPC table)
    original_cell_sweep_info_table : pd.DataFrame
        DESCRIPTION.
    original_cell_fit_table : pd.DataFrame
        DESCRIPTION.
    original_cell_feature_table : pd.DataFrame
        DESCRIPTION.
    original_Full_Adaptation_fit_table : pd.DataFrame
        DESCRIPTION.
    selection : List, optional
        List of cell item to write/modify. The default is ['All'].

    Returns
    -------
    metadata_table : pd.DataFrame
        DESCRIPTION.
    Full_TPC_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('TPC') contains a TPC pd.DataFrame (Contains Time, Current,Potential, First Time deriv of Potential, and Second Time deriv of Potential)
    Full_SF_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TPC table)
    Full_SF_dict : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TPC table)
    cell_sweep_info_table : pd.DataFrame
        Each row represents information about a sweep (Sweep nb; stim start time, stim time end, Bridge Error, ...)
    cell_fit_table : pd.DataFrame
        Contains parameters for I/O curve fit and adaptation Fit at different time-response duration
    cell_feature_table : pd.DataFrame
        Contains computed feature at different time response duration (Gain, Threshold...)
    Full_Adaptation_fit_table : pd.DataFrame
        Contains sweep-related adaptation  parameters (used for plotting)

    '''
    if 'All' in selection:
        selection = ['TPC_SF', 'Sweep_info', "Sweep_QC", 'Metadata', 'Cell_Feature_Fit']

    if database == 'Allen_Cell_Type_Database':
        cell_id = int(cell_id)
        
        
    
        
    if "TPC_SF" in selection:

        Full_TPC_table = create_cell_Full_TPC_table_concurrent_runs(
            cell_id, database)
        
        if 'Sweep_info' in selection or original_cell_sweep_info_table=='-':

            
            cell_sweep_info_table = create_cell_sweep_info_table_concurrent_runs(
                cell_id, database, Full_TPC_table.copy())
        
        else:
            cell_sweep_info_table = original_cell_sweep_info_table.copy()

        if 'Sweep_QC' in selection or original_cell_sweep_QC_table == '-':

            cell_Sweep_QC_table=create_cell_sweep_QC_table(cell_sweep_info_table)
        else:

            cell_Sweep_QC_table=original_cell_sweep_QC_table.copy()

        Full_SF_dict = create_cell_Full_SF_dict_table(
            Full_TPC_table.copy(), cell_sweep_info_table.copy())

        Full_SF_table = create_Full_SF_table(
            Full_TPC_table.copy(), Full_SF_dict.copy())
        Full_TPC_table.index = Full_TPC_table.index.astype(str)
        Full_SF_table.index = Full_SF_table.index.astype(str)

    else:

        Full_TPC_table = original_Full_TPC_table
        Full_SF_dict = original_Full_SF_dict
        Full_SF_table = original_Full_SF_table

    if "TPC_SF" not in selection and 'Sweep_info' in selection:
        Full_TPC_table = original_Full_TPC_table
        Full_SF_dict = original_Full_SF_dict
        Full_SF_table = original_Full_SF_table
        
        cell_sweep_info_table = create_cell_sweep_info_table_concurrent_runs(
            cell_id, database, Full_TPC_table.copy())
        if 'Sweep_QC' in selection:
            cell_Sweep_QC_table=create_cell_sweep_QC_table(cell_sweep_info_table)
        else:
            cell_Sweep_QC_table=original_cell_sweep_QC_table.copy()

    elif "TPC_SF" not in selection and 'Sweep_info' not in selection:

        Full_TPC_table = original_Full_TPC_table
        Full_SF_dict = original_Full_SF_dict
        Full_SF_table = original_Full_SF_table
        cell_sweep_info_table = original_cell_sweep_info_table
        if 'Sweep_QC' in selection:
            cell_Sweep_QC_table=create_cell_sweep_QC_table(cell_sweep_info_table)
        else:
            cell_Sweep_QC_table=original_cell_sweep_QC_table.copy()

    if 'Metadata' in selection:
        metadata_table = create_metadata_dict(
            cell_id, database, population_class_file, cell_sweep_info_table.copy())
    else:
        metadata_table = original_metadata_table

    if 'Cell_Feature_Fit' in selection:
        time_response_list = [.005, .010, .025, .050, .100, .250, .500]

        cell_feature_table,cell_fit_table=compute_cell_features(cell_id,Full_SF_table,cell_sweep_info_table,time_response_list)

        if cell_fit_table.shape[0] == 0:
            cell_fit_table.loc[0, :] = np.nan
            cell_fit_table = cell_fit_table.apply(pd.to_numeric)
        if cell_feature_table.shape[0] == 0:
            cell_feature_table.loc[0, :] = np.nan
            cell_feature_table = cell_feature_table.apply(pd.to_numeric)
        

    else:
        cell_fit_table = original_cell_fit_table
        cell_feature_table = original_cell_feature_table
        

    return metadata_table, Full_TPC_table, Full_SF_table, Full_SF_dict, cell_sweep_info_table, cell_Sweep_QC_table,cell_fit_table, cell_feature_table

def compute_cell_features(cell_id,Full_SF_table,cell_sweep_info_table,time_response_list):
    '''
    Compute cell I/O features as well as adaptation, for a given set of time response duration.
    If either I/O feature or adaptation can't be computed, then the function return empty dataframes,
    with justification in either I_Oobs or Adaptation_obs column of cell_fit_table.

    Parameters
    ----------
    cell_id : str or int
        
    Full_SF_table : pd.DataFrame
        DESCRIPTION.
    cell_sweep_info_table : pd.DataFrame
        DESCRIPTION.
    time_response_list : list or float
        list of time response diration to consider in s.

    Returns
    -------
    cell_feature_table : pd.DataFrame
        DESCRIPTION.
    cell_fit_table : pd.DataFrame
        DESCRIPTION.
    Full_Adaptation_fit_table : pd.DataFrame
        DESCRIPTION.

    '''
    
    fit_columns = ['Response_time_ms', 'I_O_obs', 'I_O_QNRMSE', 'Hill_amplitude',
                   'Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_sigma', 'Adaptation_obs', 'Adaptation_RMSE', 'A', 'B', 'C']
    cell_fit_table = pd.DataFrame(columns=fit_columns)

    feature_columns = ['Response_time_ms', 'Gain',
                       'Threshold', 'Saturation_Frequency',"Saturation_Stimulus", 'Adaptation_index']
    cell_feature_table = pd.DataFrame(columns=feature_columns)

    
    
    
    Full_response_stim_freq_table=fitlib.get_stim_freq_table(
        Full_SF_table.copy(), cell_sweep_info_table.copy(), .5)
    
    #max_freq_step_norm = stim_freq_max_freq_step_norm(Full_response_stim_freq_table,do_plot=False)[0]
    
    I_O_pre_obs='--'
    pruning_obs, do_fit = data_pruning_I_O(Full_response_stim_freq_table,cell_sweep_info_table)
    I_O_obs='--'
    if do_fit == True:
        I_O_obs = fitlib.fit_IO_relationship(Full_response_stim_freq_table,do_plot=False)[0]

    if do_fit == False or I_O_obs !='Hill' and I_O_obs != 'Hill-Sigmoid':
        I_O_pre_obs=str(('Cannot compute I/O,'+pruning_obs+I_O_obs))
        I_O_pre_obs.replace('--', '')
    

    adapt_obs, adapt_do_fit = data_pruning_adaptation(Full_SF_table, cell_sweep_info_table.copy(), .5)
    Adaptation_obs="--"
    Adapt_obs_reject='--'
    if adapt_do_fit == True:
        inst_freq_table = fitlib.extract_inst_freq_table(Full_SF_table, cell_sweep_info_table.copy(), .5)
        Adaptation_obs, A, Adaptation_index, C, Adaptation_RMSE = fitlib.fit_adaptation_curve(inst_freq_table, do_plot=False)
    
    if adapt_do_fit == False or Adaptation_obs!='--':
        Adapt_obs_reject='--'
        if adapt_do_fit == False or Adaptation_obs!='--':
            Adapt_obs_reject=str('Cannot compute adaptation,'+adapt_obs+Adaptation_obs)
            Adapt_obs_reject.replace('--', '')
            
    if I_O_pre_obs !='--' or Adapt_obs_reject != '--':
        
        #return cell_feature_table,cell_fit_table,Full_Adaptation_fit_table
        new_fit_table_line = pd.DataFrame([0,
                                        I_O_pre_obs,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        Adapt_obs_reject,
                                        np.nan,
                                        np.nan,
                                        np.nan,
                                        np.nan]).T
        new_fit_table_line.columns=fit_columns
        cell_fit_table=pd.concat([cell_fit_table,new_fit_table_line],ignore_index=True)
        
        cell_fit_table_convert_dict = {'Response_time_ms': float,
                        'I_O_obs':str,
                        'I_O_QNRMSE':float,
                        'Hill_amplitude':float,
                        'Hill_coef':float,
                        'Hill_Half_cst':float,
                        'Hill_x0':float,
                        'Sigmoid_x0':float,
                        'Sigmoid_sigma':float,
                        'Adaptation_obs':str,
                        'Adaptation_RMSE':float,
                        'A':float,
                        'B':float,
                        'C':float}
        cell_fit_table=cell_fit_table.astype(cell_fit_table_convert_dict)

        new_feature_table_line = pd.DataFrame([0,
                                            np.nan,
                                            np.nan,
                                            np.nan,
                                            np.nan,
                                            np.nan]).T
        new_feature_table_line.columns= feature_columns
        cell_feature_table=pd.concat([cell_feature_table,new_feature_table_line],ignore_index=True)
        
        cell_feature_table_convert_dict={'Response_time_ms': float,
                        'Gain':float,
                        'Threshold':float,
                        'Saturation_Frequency':float,
                        'Saturation_Stimulus':float,
                        'Adaptation_index':float}
        cell_feature_table=cell_feature_table.astype(cell_feature_table_convert_dict)

                                          
       
        return cell_feature_table,cell_fit_table
        
        
    ## If I_O criteria passed and adaptation was succesFully computed, compute I/O feature for different time responses

    for response_time in time_response_list:

        stim_freq_table = fitlib.get_stim_freq_table(
            Full_SF_table.copy(), cell_sweep_info_table.copy(), response_time)

        pruning_obs, do_fit = data_pruning_I_O(stim_freq_table,cell_sweep_info_table)

        if do_fit == True:
            I_O_obs, Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_sigma,I_O_QNRMSE, x_shift, Gain, Threshold, Saturation_freq,Saturation_stim = fitlib.fit_IO_relationship(
                stim_freq_table, do_plot=False)

        else:
            I_O_obs = pruning_obs
            empty_array = np.empty(12)
            empty_array[:] = np.nan
            Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_sigma,I_O_QNRMSE, x_shift, Gain, Threshold, Saturation_freq,Saturation_stim = empty_array


        if response_time == max(time_response_list):
            
            
            
            inst_freq_table = fitlib.extract_inst_freq_table(Full_SF_table, cell_sweep_info_table.copy(), response_time)
            Adaptation_obs, A, Adaptation_index, C, Adaptation_RMSE = fitlib.fit_adaptation_curve(inst_freq_table, do_plot=False)
            
        else:
            empty_array = np.empty(4)
            empty_array[:] = np.nan
            Adaptation_obs = '--'
            Adaptation_RMSE, A, Adaptation_index, C = empty_array

        new_fit_table_line = pd.Series([response_time*1e3,
                                        I_O_obs,
                                        I_O_QNRMSE,
                                        Hill_Amplitude,
                                        Hill_coef,
                                        Hill_Half_cst,
                                        Hill_x0,
                                        sigmoid_x0,
                                        sigmoid_sigma,
                                        Adaptation_obs,
                                        Adaptation_RMSE,
                                        A,
                                        Adaptation_index,
                                        C], index=fit_columns)

        cell_fit_table = cell_fit_table.append(
            new_fit_table_line, ignore_index=True)

        new_feature_table_line = pd.Series([response_time*1e3,
                                            Gain,
                                            Threshold,
                                            Saturation_freq,
                                            Saturation_stim,
                                            Adaptation_index], index=feature_columns)

        cell_feature_table = cell_feature_table.append(
            new_feature_table_line, ignore_index=True)

   
    cell_fit_table.index = cell_fit_table.loc[:, "Response_time_ms"]
    cell_feature_table.index = cell_feature_table.loc[:,
                                                      'Response_time_ms']
    cell_feature_table.index = cell_feature_table.index.astype(int)
    
    return cell_feature_table,cell_fit_table
    

def write_cell_file_h5(cell_file_path,
                       original_Full_TPC_table,
                       original_Full_SF_dict,
                       original_cell_sweep_info_table,
                       original_cell_sweep_QC,
                       original_cell_fit_table,
                       original_cell_feature_table,
                       original_metadata_table,
                       selection=['All'],
                       overwrite=False):
    '''
    This function open an existing cell file, or create if cell file doesn't already exists. It writes into the cell file the items specified in selection

    Parameters
    ----------
    cell_file_path : str
        File path to which store cell file.
    original_Full_TPC_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('TPC') contains a TPC pd.DataFrame (Contains Time, Current,Potential, First Time deriv of Potential, and Second Time deriv of Potential)
    original_Full_SF_dict : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TPC table)
    original_cell_sweep_info_table : pd.DataFrame
        Each row represents information about a sweep (Sweep nb; stim start time, stim time end, Bridge Error, ...)
    original_cell_fit_table : pd.DataFrame
        Contains parameters for I/O curve fit and adaptation Fit at different time-response duration
    original_cell_feature_table : pd.DataFrame
        Contains computed feature at different time response duration (Gain, Threshold...)
    original_Full_Adaptation_fit_table : pd.DataFrame
        Contains sweep-related adaptation  parameters (used for plotting)
    original_metadata_table : pd.DataFrame
        Contains cell metadata.
    selection : List, optional
        List of cell item to write/modify. The default is ['All'].
    overwrite : Bool, optional
        If for a given cell_id, a cell file already exists in saving_folder_path, wether to overwrite already existing information.
        The default is False.



    '''

    #f=h5py.File("/Users/julienballbe/Downloads/New_cell_file_Full_test.h5", "w")
    f = h5py.File(cell_file_path, "a")
    if 'All' in selection:
        selection = ['TPC_SF', 'Sweep_info','Sweep_QC', 'Metadata', 'Cell_Feature_Fit']
    # Store Metadata Table
    if 'Metadata' in selection:
        if 'Metadata_table' in f.keys() and overwrite == True:
            del f["Metadata_table"]
            Metadata_group = f.create_group('Metadata_table')
            for elt in original_metadata_table.keys():

                Metadata_group.create_dataset(
                    str(elt), data=original_metadata_table[elt])
        elif 'Metadata_table' not in f.keys():
            Metadata_group = f.create_group('Metadata_table')
            for elt in original_metadata_table.keys():

                Metadata_group.create_dataset(
                    str(elt), data=original_metadata_table[elt])

    # Store TPC Tables and SF dict
    if 'TPC_SF' in selection:
        Full_TPC_table = original_Full_TPC_table.copy()
        Full_SF_dict = original_Full_SF_dict.copy()
        if 'TPC_tables' in f.keys() and overwrite == True:
            del f["TPC_tables"]
            del f["SF_tables"]

            sweep_list = np.array(Full_TPC_table['Sweep'])
            TPC_group = f.create_group('TPC_tables')
            SF_group = f.create_group("SF_tables")
            TPC_colnames = np.array(
                Full_TPC_table.loc[sweep_list[0], "TPC"].columns)
            TPC_group.create_dataset("TPC_colnames", data=TPC_colnames)
            sweep_list = np.array(Full_TPC_table['Sweep'])
            for current_sweep in sweep_list:
                TPC_group.create_dataset(
                    str(current_sweep), data=Full_TPC_table.loc[current_sweep, "TPC"])

                current_SF_dict = Full_SF_dict.loc[current_sweep, "SF_dict"]
                current_SF_group = SF_group.create_group(str(current_sweep))
                for elt in current_SF_dict.keys():

                    if len(current_SF_dict[elt]) != 0:
                        current_SF_group.create_dataset(
                            str(elt), data=current_SF_dict[elt])

        elif 'TPC_tables' not in f.keys():
            sweep_list = np.array(Full_TPC_table['Sweep'])
            TPC_group = f.create_group('TPC_tables')
            SF_group = f.create_group("SF_tables")
            TPC_colnames = np.array(
                Full_TPC_table.loc[sweep_list[0], "TPC"].columns)
            TPC_group.create_dataset("TPC_colnames", data=TPC_colnames)
            for current_sweep in sweep_list:
                TPC_group.create_dataset(
                    str(current_sweep), data=Full_TPC_table.loc[current_sweep, "TPC"])

                current_SF_dict = Full_SF_dict.loc[current_sweep, "SF_dict"]
                current_SF_group = SF_group.create_group(str(current_sweep))
                for elt in current_SF_dict.keys():

                    if len(current_SF_dict[elt]) != 0:
                        current_SF_group.create_dataset(
                            str(elt), data=current_SF_dict[elt])

    # Store Cell Sweep Info Table
    if "Sweep_info" in selection:
        cell_sweep_info_table = original_cell_sweep_info_table.copy()
        if 'Sweep_info' in f.keys() and overwrite == True:
            del f["Sweep_info"]
            cell_sweep_info_table_group = f.create_group('Sweep_info')
            sweep_list=np.array(cell_sweep_info_table['Sweep'])

            for elt in np.array(cell_sweep_info_table.columns):
                cell_sweep_info_table_group.create_dataset(
                    elt, data=np.array(cell_sweep_info_table[elt]))
               
            sweep_list = np.array(cell_sweep_info_table['Sweep'])
        elif 'Sweep_info' not in f.keys():
            cell_sweep_info_table_group = f.create_group('Sweep_info')
            for elt in np.array(cell_sweep_info_table.columns):
                cell_sweep_info_table_group.create_dataset(
                    elt, data=np.array(cell_sweep_info_table[elt]))
                
            
            sweep_list = np.array(cell_sweep_info_table['Sweep'])
    
    if 'Sweep_QC' in selection:
        cell_sweep_QC = original_cell_sweep_QC.copy()
        convert_dict = {'Sweep': str,
                        'Passed_QC':bool
                        }

        cell_sweep_QC = cell_sweep_QC.astype(convert_dict)

        if 'Sweep_QC' in f.keys() and overwrite == True:
            del f['Sweep_QC']
            cell_sweep_QC_group = f.create_group("Sweep_QC")
            for elt in np.array(cell_sweep_QC.columns):
                
                cell_sweep_QC_group.create_dataset(
                    elt, data=np.array(cell_sweep_QC[elt]))
        elif 'Sweep_QC' not in f.keys():
            cell_sweep_QC_group = f.create_group("Sweep_QC")
            for elt in np.array(cell_sweep_QC.columns):
                
                cell_sweep_QC_group.create_dataset(
                    elt, data=np.array(cell_sweep_QC[elt]))
            
        
    # Store Cell Feature Table
    
    if 'Cell_Feature_Fit' in selection:
        cell_fit_table = original_cell_fit_table.copy()
        cell_feature_table = original_cell_feature_table.copy()

        if 'Cell_Feature' in f.keys() and overwrite == True:
            del f["Cell_Feature"]

            del f['Cell_Fit']


            cell_feature_group = f.create_group('Cell_Feature')
            
            cell_feature_group.create_dataset(
                'Cell_Feature_Table', data=cell_feature_table)
            
            cell_feature_group.create_dataset(
                'Cell_Feature_Table_colnames', data=np.array(cell_feature_table.columns))

            cell_fit_group = f.create_group('Cell_Fit')
            

            for elt in np.array(cell_fit_table.columns):
                cell_fit_group.create_dataset(
                    elt, data=np.array(cell_fit_table[elt]))

            

        elif 'Cell_Feature' not in f.keys():
            
            cell_feature_group = f.create_group('Cell_Feature')
            cell_feature_group.create_dataset(
                'Cell_Feature_Table', data=cell_feature_table)
            cell_feature_group.create_dataset(
                'Cell_Feature_Table_colnames', data=np.array(cell_feature_table.columns))

           

            # Store Cell fit table
            cell_fit_group = f.create_group('Cell_Fit')
            

            for elt in np.array(cell_fit_table.columns):
                cell_fit_group.create_dataset(
                    elt, data=np.array(cell_fit_table[elt]))

    f.close()


def create_cell_Full_TPC_table_concurrent_runs(cell_id, database):
    '''
    Create for a given cell Full TPC table. Calls concurrent runs with 8 cores to parralelize the process

    Parameters
    ----------
    cell_id : str or int
        Cell id.
    database : str
        Database from which the cell comes.

    Returns
    -------
    Full_TPC_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep')contains Sweep_id, and second column ('TPC') contains a TPC pd.DataFrame (Contains Time, Current,Potential, First Time deriv of Potential, and Second Time deriv of Potential)

    '''
    Full_TPC_table = pd.DataFrame(columns=['Sweep', 'TPC'])

    if database == 'Allen_Cell_Type_Database':
        ctc = CellTypesCache(
            manifest_file="/Users/julienballbe/My_Work/Allen_Data/Common_Script/Full_analysis_cell_types/manifest.json")
        my_Cell_data = ctc.get_ephys_data(cell_id)
        sweep_type = coarse_or_fine_sweep(cell_id)
        sweep_type = sweep_type[sweep_type["stimulus_description"] == "COARSE"]
        coarse_sweep_number = sweep_type.loc[:, "sweep_number"].values

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            TPC_lines_dict = {executor.submit(
                parafunc.get_sweep_TPC_parallel, x, cell_id, database, my_Cell_data): x for x in coarse_sweep_number}
            for f in concurrent.futures.as_completed(TPC_lines_dict):

                Full_TPC_table=pd.concat([Full_TPC_table,f.result()],axis=0,ignore_index=True)
                
                
        Full_TPC_table.index = Full_TPC_table.loc[:, 'Sweep']


    elif database == 'Lantyer_Database' or database == 'Lantyer_Database':

        # cell_sweep_stim_table = pd.read_pickle(
        #     '/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_Stim.pkl')
        cell_sweep_stim_table=pd.read_pickle(
            '/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_List_table_Lantyer.pkl')
        cell_sweep_stim_table = cell_sweep_stim_table[cell_sweep_stim_table['Cell_id'] == cell_id]

        sweep_list = np.array(cell_sweep_stim_table.iloc[0, -1])

        stim_start_time, stim_end_time = estimate_cell_stim_time_limit_concurrent_runs(
            cell_id, database)



        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            TPC_lines_dict = {executor.submit(
                parafunc.get_sweep_TPC_parallel, x, cell_id, database, '--', stim_start_time, stim_end_time): x for x in sweep_list}
            for f in concurrent.futures.as_completed(TPC_lines_dict):
                Full_TPC_table=pd.concat([Full_TPC_table,f.result()],axis=0,ignore_index=True)

        Full_TPC_table.index = Full_TPC_table.loc[:, 'Sweep']
        stim_start_time, stim_end_time = estimate_cell_stim_onset(
            Full_TPC_table, do_plot=False)  # estimate cell stim start/stim_end to shorten TPC

        second_Full_TPC_table = pd.DataFrame(columns=['Sweep', 'TPC'])
        for sweep in sweep_list:
            current_TPC = Full_TPC_table.loc[sweep, 'TPC'].copy()
            current_TPC = current_TPC[current_TPC['Time_s'] < (
                stim_end_time+.5)]
            current_TPC = current_TPC[current_TPC['Time_s'] > (
                stim_start_time-.5)]
            
            current_TPC = current_TPC.apply(pd.to_numeric)
            current_TPC = current_TPC.reset_index(drop=True)

            current_TPC.index = current_TPC.index.astype(int)
            new_line = pd.Series([sweep, current_TPC], index=['Sweep', 'TPC'])
            second_Full_TPC_table = second_Full_TPC_table.append(
                new_line, ignore_index=True)
        Full_TPC_table = second_Full_TPC_table
        Full_TPC_table.index = Full_TPC_table.loc[:, 'Sweep']
        
    elif database == 'NVC':
        
        cell_sweep_table=pd.read_pickle('/Volumes/Work_Julien/NVC_database/NVC_CC_Cell_sweep_table_V2.pkl')
        cell_sweep_table = cell_sweep_table[cell_sweep_table['Cell_id'] == cell_id]

        sweep_list = np.array(cell_sweep_table.iloc[0, -1])
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            TPC_lines_dict = {executor.submit(
                parafunc.get_sweep_TPC_parallel, x, cell_id, database, '--'): x for x in sweep_list}
            for f in concurrent.futures.as_completed(TPC_lines_dict):
                Full_TPC_table=pd.concat([Full_TPC_table,f.result()],axis=0,ignore_index=True)

        Full_TPC_table.index = Full_TPC_table.loc[:, 'Sweep']
        

    return Full_TPC_table





def get_traces(cell_id, sweep_id, database):
    '''
    Function to get sweep raw traces for a cell

    Parameters
    ----------
    cell_id : str or int
        DESCRIPTION.
    sweep_id : int
        DESCRIPTION.
    database : str
        DESCRIPTION.

    Returns
    -------
    time_trace : np.array
        DESCRIPTION.
    potential_trace : np.array
        DESCRIPTION.
    current_trace : np.array
        DESCRIPTION.
    stim_start_time : float
        DESCRIPTION.
    stim_end_time : TYPE
        DESCRIPTION.

    '''

    if database == 'Allen_Cell_Type_Database':
        my_Cell_data = ctc.get_ephys_data(cell_id)
        index_range = my_Cell_data.get_sweep(sweep_id)["index_range"]
        sampling_rate = my_Cell_data.get_sweep(sweep_id)["sampling_rate"]
        current_trace = (my_Cell_data.get_sweep(sweep_id)["stimulus"][0:index_range[1]+1]) * 1e12  # to pA
        potential_trace = (my_Cell_data.get_sweep(sweep_id)["response"][0:index_range[1]+1]) * 1e3  # to mV
        potential_trace = np.array(potential_trace)
        current_trace = np.array(current_trace)
        stim_start_index = index_range[0]+next( x for x, val in enumerate(current_trace[index_range[0]:]) if val != 0)
        time_trace = np.arange(0, len(current_trace)) * (1.0 / sampling_rate)
        stim_start_time = time_trace[stim_start_index]
        stim_end_time = stim_start_time+1.

    elif database == 'Lantyer_Database':

        sweep_stim_table = pd.read_pickle(
            '/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_List_table_Lantyer.pkl')

        sweep_stim_table = sweep_stim_table[sweep_stim_table['Cell_id'] == cell_id]
        file_name = sweep_stim_table.iloc[0, 0]
        #sweep_stim_table = pd.DataFrame(sweep_stim_table.iloc[0, 2])
        current_file = scipy.io.loadmat(str(
            '/Users/julienballbe/My_Work/Lantyer_Data/Original_matlab_file/'+str(file_name)))

       
        current_unique_cell_id = cell_id[-1]

        experiment,current_Protocol,current_sweep = sweep_id.split("_")
        

        current_id = str('Trace_'+current_unique_cell_id+'_' +
                         current_Protocol+'_'+str(current_sweep))

        current_stim_trace = pd.DataFrame(
            current_file[str(current_id+'_1')], columns=['Time_s', 'Input_current_pA'])
        current_stim_trace.loc[:, 'Input_current_pA'] *= 1e12

        current_membrane_trace = pd.DataFrame(
            current_file[str(current_id+'_2')], columns=['Time_s', 'Membrane_potential_mV'])
        current_membrane_trace.loc[:, 'Membrane_potential_mV'] *= 1e3

        time_trace = np.array(current_membrane_trace.loc[:, 'Time_s'])
        potential_trace = np.array(
            current_membrane_trace.loc[:, 'Membrane_potential_mV'])
        current_trace = np.array(current_stim_trace.loc[:, 'Input_current_pA'])

        
        # stim_start_time=best_stim_start
        # stim_end_time=best_stim_start+.5
        stim_start_time = np.nan
        stim_end_time = np.nan
        
    elif database == 'NVC':
        
        
        cell_file_directory = pd.read_csv('/Volumes/Work_Julien/NVC_database/Cell_files_id_directories.csv',index_col=None)
        current_table_CC = cell_file_directory[cell_file_directory['Cell_id']==cell_id ]
        current_table_CC = current_table_CC[current_table_CC['CC/DC']=='CC']
        current_table_CC = current_table_CC.sort_values(
            by=['Time'])
        current_table_CC=current_table_CC.reset_index(drop=True)
        
        # current_Protocol = int(str(sweep_id)[0])
        # current_sweep = int(str(sweep_id)[1:])
        experiment,current_Protocol,current_sweep = sweep_id.split("_")
        current_Protocol = int(current_Protocol)
        current_sweep = int(current_sweep)
        # current_experiment = int(str(sweep_id)[0])
        # current_Protocol = int(str(sweep_id)[1:3])
        # current_sweep = int(str(sweep_id)[3:])
        
        current_time_file=current_table_CC.loc[int(current_Protocol)-1,'Time']
        
        file_path=build_file_path_from_cell_id_NVC(cell_id,cell_file_directory)
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
        
        if current_sweep==len(separator_indices):
            previous_index=separator_indices[int(current_sweep)-1][0]
            raw_traces=float_array[previous_index:]
        else:
            previous_index=separator_indices[int(current_sweep)-1][0]
            index=separator_indices[current_sweep][0]
            raw_traces=float_array[previous_index:index]
            

        traces=raw_traces[1:]
        potential_current_sep_index=int(len(traces)/2)
        potential_trace=traces[:potential_current_sep_index]
        current_trace=traces[potential_current_sep_index:]
        
        sweep_duration=len(current_trace)/metadata["Sample_rate_Hz"]
        time_trace=np.arange(0,sweep_duration,(1/metadata["Sample_rate_Hz"]))
        
        stim_start_time=metadata["I_delay_ms"]*1e-3
        stim_end_time=stim_start_time+metadata["I_pulse_width_ms"]*1e-3
         
    return time_trace, potential_trace, current_trace, stim_start_time, stim_end_time


    
    
def build_file_path_from_cell_id_NVC(cell_id,cell_files_id_directories_file):
    sub_table=cell_files_id_directories_file[cell_files_id_directories_file['Cell_id']==cell_id]
    sub_table=sub_table.reset_index(drop=True)
    
    
    Full_species=sub_table.loc[0,'Species']
    Full_date=sub_table.loc[0,'Date']
    Full_cell=sub_table.loc[0,'Cell/Electrode']
    
    
    Full_file_path=Full_species+'/'+Full_date+'/'+Full_cell
    return Full_file_path
    

def create_cell_sweep_info_table_concurrent_runs(cell_id, database, cell_Full_TPC_table):
    '''
    Create cell sweep-info table

    Parameters
    ----------
    cell_id : str or int
        DESCRIPTION.
    database : str
        DESCRIPTION.
    cell_Full_TPC_table : pd.DataFrame
        Full TPC table.

    Returns
    -------
    cell_sweep_info_table : pd.DataFrame
        DataFrame with each row contains sweep-related feature.

    '''
    cell_sweep_info_table = pd.DataFrame(columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_amp_pA', 'Stim_baseline_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz'])
    sweep_list = np.array(cell_Full_TPC_table['Sweep'], dtype=str)

    if database == 'Lantyer_Database':
        stim_start_time, stim_end_time = estimate_cell_stim_onset(
            cell_Full_TPC_table, do_plot=False)
        my_Cell_data=None
    elif database == 'Allen_Cell_Type_Database':
        stim_start_time = np.nan
        stim_end_time = np.nan
        my_Cell_data = ctc.get_ephys_data(cell_id)
        
    elif database == "NVC":
        stim_start_time = np.nan
        stim_end_time = np.nan
        my_Cell_data = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        sweep_lines_dict = {executor.submit(parafunc.get_sweep_info_parallel, cell_Full_TPC_table, x,
                                            cell_id, database, stim_start_time, stim_end_time,my_Cell_data): x for x in sweep_list}

        for f in concurrent.futures.as_completed(sweep_lines_dict):
            
            cell_sweep_info_table = pd.concat([
                cell_sweep_info_table,f.result()], ignore_index=True)

    cell_sweep_info_table = cell_sweep_info_table.sort_values(
        by=['Protocol_id', 'Trace_id', 'Stim_amp_pA'])

    extrapolated_BE = pd.DataFrame(columns=['Sweep', 'Bridge_Error_GOhms'])

    cell_sweep_info_table.loc[:, 'Bridge_Error_GOhms'] = removeOutliers(
        np.array(cell_sweep_info_table.loc[:, 'Bridge_Error_GOhms']), 3)

    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'

    for elt in sweep_list:
        
        if np.isnan(cell_sweep_info_table[cell_sweep_info_table["Sweep"]==elt]["Bridge_Error_GOhms"].values[0]):
            cell_sweep_info_table[cell_sweep_info_table["Sweep"]==elt]["Bridge_Error_extrapolated"].values[0] = True
    
    
    
    Linear_table=pd.DataFrame(columns=['Sweep',"Time_constant_ms", 'Input_Resistance_MOhm', 'Membrane_resting_potential_mV','Response_Steady_State_mV'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        sweep_linear_dict = {executor.submit(ephys_treat.get_sweep_linear_properties, x,cell_Full_TPC_table, cell_sweep_info_table): x for x in sweep_list}
        for f in concurrent.futures.as_completed(sweep_linear_dict):
            Linear_table = pd.concat(
                [Linear_table,f.result()], ignore_index=True)    
    
    cell_sweep_info_table = cell_sweep_info_table.merge(
        Linear_table, how='inner', on='Sweep')
    


    for current_Protocol in cell_sweep_info_table['Protocol_id'].unique():
        reduced_cell_sweep_info_table = cell_sweep_info_table[
            cell_sweep_info_table['Protocol_id'] == current_Protocol]
        
        BE_array = np.array(
            reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
        sweep_array = np.array(reduced_cell_sweep_info_table['Sweep'])

        nan_ind_BE = np.isnan(BE_array)

        x = np.arange(len(BE_array))
        if False in nan_ind_BE:

            BE_array[nan_ind_BE] = np.interp(
                x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])

        extrapolated_BE_Series = pd.Series(BE_array)
        sweep_array = pd.Series(sweep_array)
        extrapolated_BE_Series = pd.DataFrame(
            pd.concat([sweep_array, extrapolated_BE_Series], axis=1))
        extrapolated_BE_Series.columns = ['Sweep', 'Bridge_Error_GOhms']
        extrapolated_BE = pd.concat(
            [extrapolated_BE,extrapolated_BE_Series], ignore_index=True)


    cell_sweep_info_table.pop('Bridge_Error_GOhms')

    cell_sweep_info_table = cell_sweep_info_table.merge(
        extrapolated_BE, how='inner', on='Sweep')
    


    cell_sweep_info_table = cell_sweep_info_table.loc[:, ['Sweep',
                                                          "Protocol_id",
                                                          'Trace_id',
                                                          'Stim_amp_pA',
                                                          'Stim_baseline_pA',
                                                          'Stim_start_s',
                                                          'Stim_end_s',
                                                          'Bridge_Error_GOhms',
                                                          'Bridge_Error_extrapolated',
                                                          'Time_constant_ms',
                                                          'Input_Resistance_MOhm',
                                                          'Membrane_resting_potential_mV',
                                                          "Response_Steady_State_mV",
                                                          'Sampling_Rate_Hz']]

    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'
    convert_dict = {'Sweep': str,
                    'Protocol_id': int,
                    'Trace_id': int,
                    'Stim_amp_pA': float,
                    'Stim_baseline_pA':float,
                    'Stim_start_s': float,
                    'Stim_end_s': float,
                    'Bridge_Error_GOhms': float,
                    'Bridge_Error_extrapolated': bool,
                    'Time_constant_ms': float,
                    'Input_Resistance_MOhm': float,
                    'Membrane_resting_potential_mV':float,
                    "Response_Steady_State_mV": float,
                    'Sampling_Rate_Hz': float
                    }

    cell_sweep_info_table = cell_sweep_info_table.astype(convert_dict)

    remove_outier_columns = ['Time_constant_ms', 'Input_Resistance_MOhm']

    for current_column in remove_outier_columns:
        cell_sweep_info_table.loc[:, current_column] = removeOutliers(
            np.array(cell_sweep_info_table.loc[:, current_column]), 3)

    
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'

    return cell_sweep_info_table




def create_cell_Full_SF_dict_table(original_Full_TPC_table, original_cell_sweep_info_table):
    '''
    Identify for all TPC table contained in a Full_TPC_table, all spike related_features

    Parameters
    ----------
    original_Full_TPC_table : pd.DataFrame
        DESCRIPTION.
    original_cell_sweep_info_table : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    Full_SF_dict_table : pd.DataFrame
        DESCRIPTION.

    '''
    Full_TPC_table = original_Full_TPC_table.copy()
    cell_sweep_info_table = original_cell_sweep_info_table.copy()
    sweep_list = np.array(Full_TPC_table['Sweep'])
    Full_SF_dict_table = pd.DataFrame(columns=['Sweep', "SF_dict"])

    for current_sweep in sweep_list:
        
        current_TPC=get_filtered_TPC_table(Full_TPC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)
        
        
        membrane_trace = np.array(current_TPC['Membrane_potential_mV'])
        time_trace = np.array(current_TPC['Time_s'])
        current_trace = np.array(current_TPC['Input_current_pA'])
        stim_start = cell_sweep_info_table.loc[current_sweep, 'Stim_start_s']
        stim_end = cell_sweep_info_table.loc[current_sweep, 'Stim_end_s']
        
        SF_dict = ephys_treat.identify_spike(
            membrane_trace, time_trace, current_trace, stim_start, stim_end, do_plot=False)
        new_line = pd.DataFrame([str(current_sweep), SF_dict]).T
        
        new_line.columns=['Sweep', 'SF_dict']
        Full_SF_dict_table=pd.concat([Full_SF_dict_table,new_line],ignore_index=True)
        
    Full_SF_dict_table.index = Full_SF_dict_table.loc[:, 'Sweep']
    Full_SF_dict_table.index = Full_SF_dict_table.index.astype(str)

    return Full_SF_dict_table


def create_Full_SF_table(original_Full_TPC_table, original_Full_SF_dict):
    '''
    Create for each sweep a DataFrame of spike related features corresponding to a subset of the corresponding TPC table; with Time, current and Potential values of spike related features

    Parameters
    ----------
    original_Full_TPC_table : pd.DataFrame
        DESCRIPTION.
    original_Full_SF_dict : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    Full_SF_table : pd.DataFrame
        DESCRIPTION.

    '''

    Full_TPC_table = original_Full_TPC_table.copy()
    Full_SF_dict = original_Full_SF_dict.copy()

    sweep_list = np.array(Full_TPC_table['Sweep'])
    Full_SF_table = pd.DataFrame(columns=["Sweep", "SF"])

    for current_sweep in sweep_list:

        current_sweep = str(current_sweep)

        current_TPC=get_filtered_TPC_table(Full_TPC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)
        
        current_SF_table = create_SF_table(current_TPC, Full_SF_dict.loc[current_sweep, 'SF_dict'].copy())
        new_line = pd.DataFrame(
            [str(current_sweep), current_SF_table]).T
        new_line.columns=["Sweep", "SF"]
        Full_SF_table=pd.concat([Full_SF_table,new_line],ignore_index=True)


    Full_SF_table.index = Full_SF_table['Sweep']
    Full_SF_table.index = Full_SF_table.index.astype(str)
    Full_SF_table.index.name = 'Index'

    return Full_SF_table


def data_pruning_I_O(stim_freq_table,cell_sweep_info_table):

    stim_freq_table = stim_freq_table.sort_values(
        by=['Stim_amp_pA', 'Frequency_Hz'])
    frequency_array = np.array(stim_freq_table.loc[:, 'Frequency_Hz'])
    obs = '-'
    do_fit = True
    
    ## Test if max_freq_step/max_freq >.5
    
    
    non_zero_freq = frequency_array[np.where(frequency_array > 0)]
    stim_array = np.array(stim_freq_table.loc[:, 'Stim_amp_pA'])
    stim_array_diff = np.diff(stim_array)
    freq_array_diff = np.diff(frequency_array)
    normalized_step_array = freq_array_diff/stim_array_diff
    if np.count_nonzero(frequency_array) < 4:
        obs = 'Less_than_4_response'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

    last_zero_index = np.flatnonzero(frequency_array)[0]
    if last_zero_index != 0:

        last_zero_index = np.flatnonzero(frequency_array)[0]-1
    sub_stim_array = stim_array[last_zero_index:]
    sub_freq_array = frequency_array[last_zero_index:]
    sub_stim_diff = np.diff(sub_stim_array)
    sub_freq_diff = np.diff(sub_freq_array)
    sub_step_array = sub_freq_diff/sub_stim_diff

    
    if len(np.unique(non_zero_freq)) < 3:
        obs = 'Less_than_3_different_frequencies'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

    
    # max_freq_step_norm = stim_freq_max_freq_step_norm(stim_freq_table,do_plot=False)[0]
    # if max_freq_step_norm >= .50:
    #     obs = 'max_freq_step/max_freq>=0.5'
    #     do_fit = False
    #     return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

    minimum_frequency_step = get_min_freq_step(stim_freq_table,do_plot=False)[0]
    if minimum_frequency_step >30:
        obs = 'Minimum_frequency_step_higher_than_30Hz'
        do_fit = False
        return obs, do_fit
        
    
    obs = '--'
    do_fit = True
    return obs, do_fit



def data_pruning_adaptation(original_SF_table, original_cell_sweep_info_table, response_time):
    cell_sweep_info_table = original_cell_sweep_info_table.copy()
    SF_table = original_SF_table.copy()
    maximum_nb_interval = 0
    sweep_list = np.array(SF_table.loc[:, "Sweep"])
    for current_sweep in sweep_list:

        df = pd.DataFrame(SF_table.loc[current_sweep, 'SF'])
        df = df[df['Feature'] == 'Upstroke']
        nb_spikes = (df[df['Time_s'] < (
            cell_sweep_info_table.loc[current_sweep, 'Stim_start_s']+response_time)].shape[0])

        if nb_spikes > maximum_nb_interval:
            maximum_nb_interval = nb_spikes
    if maximum_nb_interval < 4:
        do_fit = False
        adapt_obs = 'Less_than_4_intervals_at_most'
    else:
        do_fit = True
        adapt_obs = '--'
    return adapt_obs, do_fit





def coarse_or_fine_sweep(cell_id, stimulus_type="Long Square"):
    all_sweeps_table = pd.DataFrame(ctc.get_ephys_sweeps(cell_id))
    all_sweeps_table = all_sweeps_table[all_sweeps_table['stimulus_name']
                                        == stimulus_type]
    all_sweeps_table = all_sweeps_table[[
        'sweep_number', 'stimulus_description']]
    for elt in range(all_sweeps_table.shape[0]):
        if 'COARSE' in all_sweeps_table.iloc[elt, 1]:
            all_sweeps_table.iloc[elt, 1] = 'COARSE'
        elif 'FINEST' in all_sweeps_table.iloc[elt, 1]:
            all_sweeps_table.iloc[elt, 1] = 'FINEST'
    return(all_sweeps_table)


def create_raw_trace_file(cell_id):
    my_Cell_data = ctc.get_ephys_data(cell_id)
    sweep_type = coarse_or_fine_sweep(cell_id)
    sweep_type = sweep_type[sweep_type["stimulus_description"] == "COARSE"]
    coarse_sweep_number = sweep_type.loc[:, "sweep_number"].values
    raw_dataframe = pd.DataFrame(columns=[
                                 'Sweep', 'Stim_start_s', 'Stim_end_s', 'Time_array', 'Potential_array', 'Current_array'])

    for current_sweep in coarse_sweep_number:

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

        cutoff_start_idx = find_time_index(
            current_time_array, (stim_start_time-.1))
        cutoff_end_idx = find_time_index(
            current_time_array, (stim_end_time+.1))

        current_stim_array = current_stim_array[cutoff_start_idx:cutoff_end_idx]
        current_membrane_trace = current_membrane_trace[cutoff_start_idx:cutoff_end_idx]

        new_line = pd.Series([current_sweep, stim_start_time, stim_end_time, sampling_rate, current_membrane_trace, current_stim_array],
                             index=['Sweep', 'Stim_start_s', 'Stim_end_s', 'Time_array', 'Potential_array', 'Current_array'])

        raw_dataframe = raw_dataframe.append(new_line, ignore_index=True)
    raw_dataframe.to_pickle(
        '/Volumes/easystore/Raw_traces/'+str(cell_id)+'_raw_traces.pkl')


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
    assert t[0] <= t_0 <= t[-1], "Given time ({:f}) is outside of time range ({:f}, {:f})".format(
        t_0, t[0], t[-1])

    idx = np.argmin(abs(t - t_0))
    return idx


def create_cell_sweep_QC_table(cell_sweep_info_table):

    pass_sweep_QC = True
    sweep_QC_table = pd.DataFrame(columns=["Sweep", 'Passed_QC'])
    sweep_id_list = cell_sweep_info_table.loc[:, 'Sweep']
    Cell_Rin = np.nanmean(
        np.array(cell_sweep_info_table.loc[:, 'Input_Resistance_MOhm']))
    Cell_Rin *= 1e-3  # convert Input resistance to GOhms

    for sweep in sweep_id_list:

        ### Absolute value of Bridge Error must be lower than 0.5*Rin ###
        current_BE = cell_sweep_info_table.loc[sweep, 'Bridge_Error_GOhms']

        if np.abs(current_BE) > .5*Cell_Rin:
            pass_sweep_QC = False

        sweep_line = pd.DataFrame([str(sweep), pass_sweep_QC]).T
        
        sweep_line.columns=["Sweep", 'Passed_QC']
        sweep_QC_table=pd.concat([sweep_QC_table,sweep_line],ignore_index=True)
        

    return sweep_QC_table


def create_SF_table(original_TPC_table, SF_dict):
    TPC_table = original_TPC_table.copy()
    feature_table = pd.DataFrame(columns=['Time_s', 'Membrane_potential_mV', 'Input_current_pA',
                                 'Potential_first_time_derivative_mV/s', 'Potential_second_time_derivative_mV/s/s', 'Feature'])
    # TPC_table['Membrane_potential_mV']=np.array(filter_trace(TPC_table['Membrane_potential_mV'],
    #                                                          TPC_table['Time_s'],filter=5.,do_plot=False))
    for feature in SF_dict.keys():
        
        current_feature_table = TPC_table.loc[SF_dict[feature], :].copy()
        current_feature_table['Feature'] = feature
        feature_table = pd.concat([feature_table, current_feature_table])

    return feature_table




def create_metadata_dict(cell_id, database, population_class, original_cell_sweep_info_table):
    '''
    Create for a given cell Metadata dict

    Parameters
    ----------
    cell_id : str or int
        DESCRIPTION.
    database : str
        DESCRIPTION.
    population_class : pd.DataFrame
        DESCRIPTION.
    original_cell_sweep_info_table : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    metadata_table_dict : Dict
        DESCRIPTION.

    '''
    metadata_table = pd.DataFrame(columns=["Database",
                                           "Area",
                                           "Layer",
                                           "Dendrite_type",
                                           'Nb_of_Traces',
                                           'Nb_of_Protocol'])

    population_class.index = population_class.loc[:, "Cell_id"]
    cell_sweep_info_table = original_cell_sweep_info_table.copy()
    if database == "Allen_Cell_Type_Database":
        cell_data = ctc.get_ephys_features(cell_id)
        cell_data = cell_data[cell_data["specimen_id"] == cell_id]
        cell_data.index = cell_data.loc[:, 'specimen_id']

        population_class_line = population_class[population_class["Cell_id"] == str(
            cell_id)]
        Area = population_class.loc[str(cell_id), "General_area"]
        Layer = population_class_line.loc[str(cell_id), "Layer"]
        dendrite_type = population_class_line.loc[str(
            cell_id), "Dendrite_type"]
        Nb_of_Traces = cell_sweep_info_table.shape[0]
        Nb_of_Protocol = 1

    elif database == 'Lantyer_Database':
        population_class_line = population_class[population_class["Cell_id"] == str(
            cell_id)]
        Area = population_class.loc[str(cell_id), "General_area"]
        Layer = population_class_line.loc[str(cell_id), "Layer"]
        dendrite_type = population_class_line.loc[str(
            cell_id), "Dendrite_type"]

        Nb_of_Protocol = len(cell_sweep_info_table['Protocol_id'].unique())
        Nb_of_Traces = cell_sweep_info_table.shape[0]
        
    elif database == 'NVC':
        population_class_line = population_class[population_class["Cell_id"] == str(
            cell_id)]
        Area = population_class.loc[str(cell_id), "General_area"]
        Layer = population_class_line.loc[str(cell_id), "Layer"]
        dendrite_type = population_class_line.loc[str(
            cell_id), "Dendrite_type"]
        
        Nb_of_Protocol = len(cell_sweep_info_table['Protocol_id'].unique())
        Nb_of_Traces = cell_sweep_info_table.shape[0]
        

    new_line = pd.DataFrame([database,
                          Area,
                          Layer,
                          dendrite_type,
                          Nb_of_Traces,
                          Nb_of_Protocol]).T
    new_line.columns=["Database",
                      "Area",
                      "Layer",
                      "Dendrite_type",
                      'Nb_of_Traces',
                      'Nb_of_Protocol']
    metadata_table=pd.concat([metadata_table,new_line],ignore_index=True)

    metadata_table_dict = metadata_table.loc[0, :].to_dict()
    return metadata_table_dict


def create_cell_json_file(cell_id, metadata_table, Full_TPC_table, Full_SF_table, Full_SF_dict, cell_sweep_info_table, cell_fit_table, cell_feature_table, Full_Adaptation_fit_table):
    my_dict = {'Metadata': [metadata_table],
               'TPC': [Full_TPC_table],
               'Spike_feature_table': [Full_SF_table],
               'Fit_table': [cell_fit_table],
               'IO_table': [cell_feature_table]}

    my_dict = pd.DataFrame(my_dict)
    my_dict.to_json(
        str('/Users/julienballbe/Downloads/Data_test_'+str(cell_id)+'.json'))


# def get_TPC_SF_table_concurrent_runs_bis(file, sweep):

#     current_file = h5py.File(file, 'r')
#     TPC_group = current_file['TPC_tables']
#     current_TPC_table = TPC_group[str(sweep)]
    
#     # TPC_colnames = np.array([x.decode('ascii')
#     #                         for x in list(TPC_group['TPC_colnames'])], dtype='str')
#     TPC_colnames = list(TPC_group['TPC_colnames'])
#     current_TPC_table = pd.DataFrame(current_TPC_table)
#     current_TPC_table.columns = TPC_colnames
    
#     new_line_TPC = pd.Series(
#         [str(sweep), current_TPC_table], index=['Sweep', 'TPC'])
    
    
#     SF_group = current_file['SF_tables']
#     current_SF_table = SF_group[str(sweep)]
#     SF_dict = {}

#     for feature in current_SF_table.keys():
#         SF_dict[feature] = np.array(current_SF_table[feature])

#     new_line_SF = pd.Series([str(sweep), SF_dict], index=['Sweep', 'SF_dict'])

#     return new_line_TPC, new_line_SF


def import_h5_file(file_path, selection=['All']):

    current_file=h5py.File('/Volumes/Work_Julien/Cell_Data_File/Cell_170508AL258_1_data_file.h5', 'r')
    current_file = h5py.File(file_path, 'r')

    Full_TPC_table = pd.DataFrame()
    Full_SF_dict_table = pd.DataFrame()
    Full_SF_table = pd.DataFrame()
    Metadata_table = pd.DataFrame()
    sweep_info_table = pd.DataFrame()
    Sweep_QC_table = pd.DataFrame()
    cell_fit_table = pd.DataFrame()
    cell_feature_table = pd.DataFrame()


    if 'All' in selection:
        selection = ['TPC_SF', 'Sweep_info','Sweep_QC', 'Metadata', 'Cell_Feature_Fit']

    if 'TPC_SF' in selection and 'TPC_tables' in current_file.keys():
        ## TPC and SF ##

        TPC_group = current_file['TPC_tables']
        SF_group = current_file['SF_tables']
        Full_TPC_table = pd.DataFrame(columns=['Sweep', 'TPC'])
        sweep_list = list(TPC_group.keys())
        sweep_list.remove('TPC_colnames')
        sweep_list = np.array(sweep_list)

        Full_SF_dict_table = pd.DataFrame(columns=['Sweep', 'SF_dict'])

        with Pool() as pool:
            params = [(file_path, x) for x in sweep_list]

            for result in pool.starmap(parafunc.get_TPC_SF_table_concurrent_runs_bis, params, chunksize=(1)):
                
                Full_TPC_table = pd.concat([Full_TPC_table,
                    result[0]], ignore_index=True)
                Full_SF_dict_table = pd.concat([Full_SF_dict_table,
                    result[1]], ignore_index=True)
        
        
        Full_TPC_table.index = Full_TPC_table["Sweep"]
        Full_TPC_table.index = Full_TPC_table.index.astype(str)

        Full_SF_dict_table.index = Full_SF_dict_table["Sweep"]
        Full_SF_dict_table.index = Full_SF_dict_table.index.astype(str)

        Full_SF_table = create_Full_SF_table(
            Full_TPC_table, Full_SF_dict_table)

    elif 'TPC_SF' in selection and 'TPC_tables' not in current_file.keys():
        print('File does not contain TPC_SF group')

    if 'Metadata' in selection and 'Metadata_table' in current_file.keys():
        ## Metadata ##

        Metadata_group = current_file['Metadata_table']
        Metadata_dict = {}
        for data in Metadata_group.keys():
            if type(Metadata_group[data][()]) == bytes:
                Metadata_dict[data] = Metadata_group[data][()].decode('ascii')
            else:
                Metadata_dict[data] = Metadata_group[data][()]
        Metadata_table = pd.DataFrame(Metadata_dict, index=[0])

    elif 'Metadata' in selection and 'Metadata_table' not in current_file.keys():
        print('File does not contains Metadata group')

    if 'Sweep_info' in selection and 'Sweep_info' in current_file.keys():

        ## Sweep_info_table ##

        Sweep_info_group = current_file['Sweep_info']
        Sweep_info_dict = {}
        for data in Sweep_info_group.keys():
            Sweep_info_dict[data] = Sweep_info_group[data][()]
        Sweep_info_dict['Sweep']=Sweep_info_dict['Sweep'].astype(str)
        sweep_info_table = pd.DataFrame(
            Sweep_info_dict, index=Sweep_info_dict['Sweep'])

    elif 'Sweep_info' in selection and 'Sweep_info' not in current_file.keys():
        print('File does not contains Sweep_info group')
        
    if 'Sweep_QC' in selection and 'Sweep_QC' in current_file.keys():

        ## Sweep_QC_table ##

        Sweep_QC_group = current_file['Sweep_QC']
        Sweep_QC_dict = {}
        for data in Sweep_QC_group.keys():
            Sweep_QC_dict[data] = Sweep_QC_group[data][()]

        Sweep_QC_table = pd.DataFrame(
            Sweep_QC_dict, index=Sweep_QC_dict['Sweep'])

    elif 'Sweep_QC' in selection and 'Sweep_QC' not in current_file.keys():
        print('File does not contains Sweep_QC group')
    

    if 'Cell_Feature_Fit' in selection and 'Cell_Fit' in current_file.keys():
        ## Cell_fit ##

        Cell_fit_group = current_file['Cell_Fit']
        Cell_fit_dict = {}
        for data in Cell_fit_group.keys():
            if type(Cell_fit_group[data][(0)]) == bytes:
                Cell_fit_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_fit_group[data][()]], dtype='str')
            else:
                Cell_fit_dict[data] = Cell_fit_group[data][()]

        cell_fit_table = pd.DataFrame(
            Cell_fit_dict, index=Cell_fit_dict['Response_time_ms'])
        cell_fit_table.index = cell_fit_table.index.astype(int)

        ## Cell_feature ##

        Cell_feature_group = current_file['Cell_Feature']
        cell_feature_table = pd.DataFrame(
            Cell_feature_group['Cell_Feature_Table'])
        cell_feature_table.columns = np.array([x.decode(
            'ascii') for x in Cell_feature_group['Cell_Feature_Table_colnames'][()]], dtype='str')
        cell_feature_table.index = cell_feature_table['Response_time_ms']
        cell_feature_table.index = cell_feature_table.index.astype(int)

        
    elif 'Cell_Feature_Fit' in selection and 'Cell_Fit' not in current_file.keys():
        print('File does not contains Cell_Fit group')

    current_file.close()
    return Full_TPC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table


# def filter_trace(table, cutoff_freq, do_plot=False):
#     '''
#     Apply a low pass filter on the stimulus trace 

#     Parameters
#     ----------
#     table : DataFrame
#         Stimulus trace table, 2 columns
#         First column = 'Time_s'
#         Second column = 'Input_current_pA'
#     cutoff_freq : float
#         Cut off frequency to use in the low pass filter in Hz.
#     do_plot : Bool, optional
#         The default is False.

#     Returns
#     -------
#     table : DataFrame
#         Stimulus trace table, 3 columns
#         First column = 'Time_s'
#         Second column = 'Input_current_pA'.
#         Third column = "Filtered_Stimulus_trace_pA" --> filtered stimulus trace

#     '''

#     sampling_frequency = table.shape[0]/table.iloc[-1, 0]  # Hz

#     b, a = scipy.signal.butter(3, Wn=cutoff_freq, fs=sampling_frequency)

#     table['Filtered_Stimulus_trace_pA'] = scipy.signal.filtfilt(
#         b, a, table.iloc[:, 1])

#     if do_plot == True:
#         myplot = ggplot(table, aes(x=table.iloc[:, 0], y=table.iloc[:, 1]))+geom_line(
#             color='blue')+geom_line(table, aes(x=table.iloc[:, 0], y=table.iloc[:, 2]), color='red')
#         myplot += ylim(0, 150)
#         print(myplot)

#     return table


def estimate_cell_stim_time_limit_concurrent_runs(cell_id, database):

    if database == "Lantyer_Database":

        cell_sweep_stim_table=pd.read_pickle(
            '/Users/julienballbe/My_Work/Lantyer_Data/Cell_Sweep_List_table_Lantyer.pkl')
        cell_sweep_stim_table = cell_sweep_stim_table[cell_sweep_stim_table['Cell_id'] == cell_id]
        sweep_list = np.array(cell_sweep_stim_table.iloc[0, -1])

    Cell_stim_time_limit = pd.DataFrame(
        columns=['Sweep', 'Stim_start_s', 'Stim_end_s'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        sweep_lines_dict = {executor.submit(
            parafunc.estimate_sweep_stim_time_limit_parallel, cell_id, x, database): x for x in sweep_list}
        for f in concurrent.futures.as_completed(sweep_lines_dict):
            Cell_stim_time_limit = Cell_stim_time_limit.append(
                f.result(), ignore_index=True)

    cell_stim_start = Cell_stim_time_limit['Stim_start_s'].mode()[0]
    cell_stim_end = Cell_stim_time_limit['Stim_end_s'].mode()[0]

    return cell_stim_start, cell_stim_end





def estimate_cell_stim_onset(Full_TPC_table_original, do_plot=False):
    '''
    Compute from stimulus trace table the stimulus onset, and its duration by using autocorrelation

    Parameters
    ----------
    stim_table : DataFrame
        Stimulus trace table, 2 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'

    Returns
    -------
    best_autocorr : float
        lowest autocorrelation coefficient encontered
    best_stim_start : float
        Stimulus onset time point estimated from lowest autocorrelation coef (in s)
    best_time_autocorr : float
        Stimulus duration estimated from time shift used for best autocorrelation coefficient (in s)

    '''

    Full_TPC_table = Full_TPC_table_original.copy()
    stim_time_table = pd.DataFrame(
        columns=['Sweep', 'Stim_start_s', 'Stim_end_s'])
    for sweep in np.array(Full_TPC_table.loc[:, 'Sweep']):
        
        stim_table=get_filtered_TPC_table(Full_TPC_table,sweep,do_filter=True,filter=5.,do_plot=False)
        
        current_derivative = get_derivative(np.array(stim_table['Input_current_pA']),
                                            np.array(stim_table['Time_s']))
        current_derivative=np.insert(current_derivative,0,np.nan)
        stim_table["Filtered_Stimulus_trace_derivative_pA/ms"]=np.array(current_derivative)
        # remove last 50ms of signal (potential step)
        limit = stim_table.shape[0] - \
            int(0.05/(stim_table.iloc[1, 0]-stim_table.iloc[0, 0]))

        stim_table.loc[limit:,
                       "Filtered_Stimulus_trace_derivative_pA/ms"] = np.nan
        
        stim_table = get_autocorrelation(stim_table, .500, do_plot=do_plot)
        # best_time_autocorr=.500
        #best_stim_start_index=next(x for x, val in enumerate(stim_table['Autocorrelation']) if np.abs(val) >= 10000 )
        # best_stim_start=stim_table.iloc[best_stim_start_index,0]
        best_stim_start = stim_table[stim_table['Autocorrelation'] == np.nanmin(
            stim_table['Autocorrelation'])].iloc[0, 0]

        stim_end = best_stim_start+.500
        new_line = pd.Series([sweep, best_stim_start, stim_end], index=[
                             'Sweep', 'Stim_start_s', 'Stim_end_s'])
        stim_time_table = stim_time_table.append(new_line, ignore_index=True)


    cell_stim_start = stim_time_table['Stim_start_s'].mode()[0]
    cell_stim_end = stim_time_table['Stim_end_s'].mode()[0]
    return cell_stim_start, cell_stim_end


def get_autocorrelation(table, time_shift, do_plot=False):
    '''
    Compute autocorrelation at each time point for a stimulus trace table

    Parameters
    ----------
    table : DataFrame
        Stimulus trace table, 3 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
    time_shift : float
        Time shift in s .
    do_plot : Bool, optional
        The default is False.

    Returns
    -------
    table : DataFrame
        Stimulus trace table, 6 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
        4th column = "Filtered_Stimulus_trace_derivative_pA/ms"
        5th column = "Shifted_trace" --> Filtered stiumulus trace derivatove shifted by 'time_shift'
        6th column = 'Autocorrelation --> Autocorrelation between 4th and 5th column
    '''

    shift = int(time_shift/(table.iloc[1, 0]-table.iloc[0, 0]))

    table["Shifted_trace"] = table['Filtered_Stimulus_trace_derivative_pA/ms'].shift(
        -shift)

    table['Autocorrelation'] = table['Filtered_Stimulus_trace_derivative_pA/ms'] * \
        table["Shifted_trace"]

    if do_plot == True:

        myplot = ggplot(table, aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Filtered_Stimulus_trace_derivative_pA/ms']))+geom_line(
            color='blue')+geom_line(table, aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Autocorrelation']), color='red')

        myplot += xlab(str("Time_s; Time_shift="+str(time_shift)))
        print(myplot)

    return table




def removeOutliers(x, outlierConstant=3):
    a = np.array(x)

    
    upper_quartile = np.nanpercentile(a, 75)
    lower_quartile = np.nanpercentile(a, 25)

    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
        else:
            resultList.append(np.nan)

    return np.array(resultList)


def import_cell_json_file(file_path):
    Full_json_file = pd.read_json(file_path)
    Full_TPC_table = pd.DataFrame(Full_json_file["TPC"][0])
    Full_TPC_table.index = Full_TPC_table.index.astype(str)

    Full_SF_table = pd.DataFrame(Full_json_file["Spike_feature_table"][0])
    Full_SF_table.index = Full_SF_table.index.astype(str)

    Metadata_table = pd.DataFrame(Full_json_file["Metadata"][0])
    Fit_table = pd.DataFrame(Full_json_file["Fit_table"][0])
    IO_table = pd.DataFrame(Full_json_file["IO_table"][0])
    Adaptation_fit_table = pd.DataFrame(
        Full_json_file['Adaptation_fit_table'][0])

    return Full_SF_table, Full_TPC_table, Metadata_table, Fit_table, IO_table, Adaptation_fit_table


def import_cell_pkl_file(file_path):
    Full_pkl_file = pd.read_pickle(file_path)
    Full_TPC_table = pd.DataFrame(Full_pkl_file["TPC"][0])
    Full_TPC_table.index = Full_TPC_table.index.astype(str)

    Full_SF_table = pd.DataFrame(Full_pkl_file["Spike_feature_table"][0])
    Full_SF_table.index = Full_SF_table.index.astype(str)

    Metadata_table = pd.DataFrame(Full_pkl_file["Metadata"][0])
    Fit_table = pd.DataFrame(Full_pkl_file["Fit_table"][0])
    IO_table = pd.DataFrame(Full_pkl_file["IO_table"][0])
    Adaptation_fit_table = pd.DataFrame(
        Full_pkl_file['Adaptation_fit_table'][0])

    return Full_SF_table, Full_TPC_table, Metadata_table, Fit_table, IO_table, Adaptation_fit_table


def import_cell_hdf_file(file_path):
    Full_hdf_file = pd.read_hdf(file_path, 'df')
    Full_TPC_table = pd.DataFrame(Full_hdf_file["TPC"][0])
    Full_TPC_table.index = Full_TPC_table.index.astype(str)

    Full_SF_table = pd.DataFrame(Full_hdf_file["Spike_feature_table"][0])
    Full_SF_table.index = Full_SF_table.index.astype(str)

    Metadata_table = pd.DataFrame(Full_hdf_file["Metadata"][0])
    Fit_table = pd.DataFrame(Full_hdf_file["Fit_table"][0])
    IO_table = pd.DataFrame(Full_hdf_file["IO_table"][0])
    Adaptation_fit_table = pd.DataFrame(
        Full_hdf_file['Adaptation_fit_table'][0])

    return Full_SF_table, Full_TPC_table, Metadata_table, Fit_table, IO_table, Adaptation_fit_table


def stim_freq_max_freq_step_norm(stim_freq_table,width=None,do_plot=False):
    stim_freq_table=stim_freq_table.sort_values(by=['Stim_amp_pA',"Frequency_Hz"])
    
    # determine bin_width based on the data
    # get 0.75 quantile of stim_diff array
    
    if width == None:
        stim_diff=np.diff(np.array(stim_freq_table['Stim_amp_pA']))
        stim_diff_quantile = np.quantile(stim_diff,.75)
    
        # define bin width as 1.5* stim_diff_quantile
    
        width = np.round(stim_diff_quantile * 1.5)
        

    
    starting_min_bin_edge = math.floor(min(stim_freq_table["Stim_amp_pA"]/ 100.00)) * 100 -100
    
    max_diff_array = []
    
    for elt in range (11):
        shift = elt * (width/10)
        
    
    
    
        #n_bins = int((max(stim_freq_table["Stim_amp_pA"])-min(stim_freq_table["Stim_amp_pA"])) // width)
        #print(starting_min_bin_edge,shift,elt,width)
        #bin_edges = np.linspace(math.floor(min(stim_freq_table["Stim_amp_pA"]/ 100.00)) * 100, max(stim_freq_table["Stim_amp_pA"]), n_bins + 1)
        #bin_edges = np.linspace(starting_min_bin_edge+shift, max(stim_freq_table["Stim_amp_pA"]), n_bins + 1)
        bin_edges = np.arange(starting_min_bin_edge+shift, max(stim_freq_table["Stim_amp_pA"])+2*width, step = width)
        #print(bin_edges)
        current_output,current_bin_edges,current_bins_number = scipy.stats.binned_statistic(stim_freq_table["Stim_amp_pA"],
                                        stim_freq_table["Frequency_Hz"],
                                        statistic='mean',
                                        bins=bin_edges)
        
        
        
        
        #current_output=current_output[~np.isnan(current_output)]
        
        
        
        if do_plot:# and max_diff_array[-1]==np.nanmax(max_diff_array):
            
            my_plot=ggplot(stim_freq_table,aes(x='Stim_amp_pA',y='Frequency_Hz'))+geom_point(alpha=.4)
            bin_center_array=[]
            for elt in np.arange(1,len(current_bin_edges)):
                bin_center_array.append((current_bin_edges[elt]+current_bin_edges[elt-1])/2)
                
            psth_df=pd.DataFrame({'Bin_index':np.array(bin_center_array),
                                  'Frequency_Hz':current_output})
            
            my_plot+=geom_point(psth_df,aes(x="Bin_index",y="Frequency_Hz"),shape='_',size=width//4,stroke=.8)
            my_plot+=geom_vline(xintercept=bin_edges,color='red')
            my_plot+=ggtitle('Binned_Spike_Frequency')
            
            my_plot += xlim(starting_min_bin_edge+100, int(math.ceil(max(stim_freq_table["Stim_amp_pA"] / 100.0)) * 100))
            
            print(my_plot)
    
        sub_current_output=current_output[~np.isnan(current_output)]
        freq_step=np.diff(sub_current_output)
        max_step_index = np.nanargmax(freq_step)

        
        if max_step_index != len(freq_step)-1 :

            if freq_step[max_step_index+1]/freq_step[max_step_index] < -.5:

                continue
        max_diff_array.append(np.nanmax(np.diff(sub_current_output)))
        
        if  max_diff_array[-1]==np.nanmax(max_diff_array):
            max_freq_step_plot=ggplot(stim_freq_table,aes(x='Stim_amp_pA',y='Frequency_Hz'))+geom_point(alpha=.4)
            bin_center_array=[]
            for elt in np.arange(1,len(current_bin_edges)):
                bin_center_array.append((current_bin_edges[elt]+current_bin_edges[elt-1])/2)
                
            psth_df=pd.DataFrame({'Bin_index':np.array(bin_center_array),
                                  'Frequency_Hz':current_output})
            
            max_freq_step_plot+=geom_point(psth_df,aes(x="Bin_index",y="Frequency_Hz"),shape='_',size=width//4,stroke=.8)
            max_freq_step_plot+=geom_vline(xintercept=bin_edges,color='red')
            max_freq_step_plot+=ggtitle('Maximum_frequency_step')
            max_freq_step_plot += xlim(starting_min_bin_edge+100, int(math.ceil(max(stim_freq_table["Stim_amp_pA"] / 100.0)) * 100))
            
            
    if do_plot:
        print(max_freq_step_plot)        

    return np.nanmax(max_diff_array)/np.nanmax(stim_freq_table["Frequency_Hz"]) ,np.nanmax(max_diff_array), width,(width/10)
    

def has_fixed_dt(t): 
    """Check that all time intervals are identical."""
    dt = np.diff(t)
    
    return np.allclose(dt, np.ones_like(dt) * dt[0])

def filter_trace(value_trace, time_trace, filter=5., filter_order = 2, do_plot=False): 
    """Low-pass filters time-varying signal.

    Parameters
    ----------
    value_trace : numpy array of voltage time series in mV
    time_trace : numpy array of times in seconds
    filter : cutoff frequency for 2nd order Butterworth filter
    filter_order : order of the filter to apply
    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """
    


    delta_t = time_trace[1] - time_trace[0]
    sample_freq = 1. / delta_t

    filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency

    if filt_coeff < 0 or filt_coeff >= 1:
        raise ValueError("Butterworth coeff ({:f}) is outside of valid range [0,1]; cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
    b, a = scipy.signal.butter(filter_order, filt_coeff, "low")
    
    zi = scipy.signal.lfilter_zi(b, a)
    
    filtered_signal =  scipy.signal.lfilter(b, a, value_trace,zi=zi*value_trace[0], axis=0)[0]
    
    if do_plot:
        signal_df = pd.DataFrame({'Time_s':np.array(time_trace),
                                  'Values':np.array(value_trace)})
        filtered_df = pd.DataFrame({'Time_s':np.array(time_trace),
                                  'Values':np.array(filtered_signal)})
        
        signal_df ['Trace']='Original_Trace'
        filtered_df ['Trace']='Filtered_Trace'
        
        signal_df=signal_df.append(filtered_df,ignore_index=True)
        
        filter_plot = ggplot(signal_df, aes(x='Time_s',y='Values',color='Trace',group='Trace'))+geom_line()
        print(filter_plot)
        
        
        
    
    return filtered_signal
    
def get_derivative(value_trace, time_trace):
    
    dv = np.diff(value_trace)

    dt = np.diff(time_trace)
    dvdt = 1e-3 * dv / dt # in V/s = mV/ms

    # Remove nan values (in case any dt values == 0)
    dvdt = dvdt[~np.isnan(dvdt)]

    return dvdt


def get_filtered_TPC_table(original_cell_full_TPC_table,sweep,do_filter=True,filter=5.,do_plot=False):
    '''
    

    Parameters
    ----------
    original_cell_full_TPC_table : pd.DataFrame
        Table containing all cell's sweep-TPC tables.
    sweep : int
        Sweep id.
    do_filter : Bool, optional
        Filter Membrane potential and Current traces. The default is True.
    do_plot : Bool, optional
       The default is False.

    Returns
    -------
    TPC_table : pd.DataFrame
        Single Sweep TPC table with filtered (or not) membrane potential and current traces.

    '''
    
    cell_full_TPC_table = original_cell_full_TPC_table.copy()
    TPC_table=cell_full_TPC_table.loc[sweep,'TPC'].copy()
    
    if do_filter:

        TPC_table['Membrane_potential_mV']=np.array(filter_trace(TPC_table['Membrane_potential_mV'],
                                                                            TPC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
        
        TPC_table['Input_current_pA']=np.array(filter_trace(TPC_table['Input_current_pA'],
                                                                            TPC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
        
    if do_plot:
        potential_plot = ggplot(TPC_table,aes(x='Time_s',y='Membrane_potential_mV'))+geom_line()
        potential_plot+=ggtitle(str('Sweep:'+str(sweep)+'Membrane_Potential_mV'))
        print(potential_plot)
        current_plot=ggplot(TPC_table,aes(x='Time_s',y='Input_current_pA'))+geom_line()
        current_plot+=ggtitle(str('Sweep:'+str(sweep)+'Input_current_pA'))
        print(current_plot)


    return TPC_table


def get_min_freq_step(original_stimulus_frequency_table,do_plot=False):
    
    original_data_table =  original_stimulus_frequency_table.copy()
   
    original_data = original_data_table.copy()
    
    original_data = original_data.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    original_x_data = np.array(original_data['Stim_amp_pA'])
    original_y_data = np.array(original_data['Frequency_Hz'])
    extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
    
    first_stimulus_frequency_table=original_data_table.copy()
    first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    try:
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
        
       
    
        third_order_poly_model = PolynomialModel(degree = 3)
        pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        
        if len(trimmed_y_data)>=4:
            
            third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
        
            best_c0 = third_order_poly_model_results.best_values['c0']
            best_c1 = third_order_poly_model_results.best_values['c1']
            best_c2 = third_order_poly_model_results.best_values['c2']
            best_c3 = third_order_poly_model_results.best_values['c3']
            
            delta_3rd_order_poly = 4*(best_c2**2)-4*3*best_c3*best_c1 #check if 3rd order polynomial reaches local max/minimum
            
            
        
            extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            
            extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
            trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_3rd_poly_model})
            trimmed_3rd_poly_table['Legend']='3rd_order_poly'
            my_plot=ggplot(first_stimulus_frequency_table,aes(x='Stim_amp_pA',y='Frequency_Hz'))+geom_point()
            my_plot+=geom_line(trimmed_3rd_poly_table,aes(x='Stim_amp_pA',y='Frequency_Hz'))
            if do_plot:
                
                print(my_plot)
            
            zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
           
          
            
            if len(zero_crossings_3rd_poly_model_freq_diff)==0:
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table.copy()
                
            elif len(zero_crossings_3rd_poly_model_freq_diff)==1:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                    
            elif len(zero_crossings_3rd_poly_model_freq_diff)==2:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
                
                
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
        
        else:
            Descending_segment=False
                
                
                
                
                
            
       
            ### 1 - end
            

        
        ### 2 - Trim polynomial fit; keep [mean-0.5*poly_amplitude ; [mean+0.5*poly_amplitude]
        if Descending_segment:
            
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
            descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
            
            max_freq_poly_fit=np.nanmax(trimmed_3rd_poly_table['Frequency_Hz'])
            trimmed_ascending_segment['Legend']="Trimmed_asc_seg"
            trimmed_descending_segment['Legend']="Trimmed_desc_seg"
           
        
        else:
            #end_slope_positive --> go for 2nd order polynomial
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
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            max_freq_poly_fit=np.nanmax(trimmed_2nd_poly_table['Frequency_Hz'])
            
            trimmed_2nd_poly_table['Legend']='2nd_order_poly'
            trimmed_ascending_segment['Legend']="Trimmed_asc_seg"
            
        ### 2 - end 
        
        ### 3 - Linear fit on polynomial trimmed data
        ascending_linear_slope_init,ascending_linear_intercept_init=fitlib.fit_specimen_fi_slope(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        if Descending_segment:
            #ascending_sigmoid_fit += Model(sigmoid_amplitude_function)
            descending_linear_slope_init,descending_linear_intercept_init=fitlib.fit_specimen_fi_slope(trimmed_descending_segment["Stim_amp_pA"],
                                                  trimmed_descending_segment['Frequency_Hz'])
        
        
        
        
        ### 3 - end
        
        ### 4 - Fit single or double Sigmoid
        
        ascending_sigmoid_fit= Model(fitlib.sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        
        
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        if Descending_segment:
            #ascending_sigmoid_fit += Model(sigmoid_amplitude_function)
            
            
           
            descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
            
            descending_sigmoid_fit = Model(fitlib.sigmoid_function,prefix='desc_')
            descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
            descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
            descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
            
            ascending_sigmoid_fit *=descending_sigmoid_fit
            ascending_segment_fit_params+=descending_sigmoid_fit_pars

        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        asc_sigmoid_fit = best_value_Amp * fitlib.sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
        #asc_sigmoid_fit = sigmoid_amplitude_function(original_x_data , best_value_A_asc, best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit = best_value_Amp * fitlib.sigmoid_function(extended_x_data , best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        
       
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        
        if Descending_segment:
            
            best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
            best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
            
            full_sigmoid_fit = best_value_Amp * fitlib.sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * fitlib.sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
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
            
           
           
            
            sigmoid_fit_plot =  ggplot(original_data,aes(x='Stim_amp_pA',y="Frequency_Hz"))+geom_point()
            sigmoid_fit_plot += geom_line(full_sigmoid_fit_table, aes(x='Stim_amp_pA',y="Frequency_Hz"))
            #sigmoid_fit_plot += geom_line(trimmed_ascending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += ggtitle("Sigmoid_fit_to original_data")
            
            # if Descending_segment:
                
                
            #     sigmoid_fit_plot += geom_line(trimmed_descending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
                
            
           
            
            
    
            print(sigmoid_fit_plot)
         ### 4 - end

        maximum_fit_frequency=np.nanmax(full_sigmoid_fit)
        maximum_fit_frequency_index = np.nanargmax(full_sigmoid_fit)
        maximum_fit_stimulus = extended_x_data[maximum_fit_frequency_index]
        original_data_table = original_data_table.sort_values(by=['Stim_amp_pA'])
        
        noisy_spike_freq_threshold = np.nanmax([.04*maximum_fit_frequency,2.])
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        

        
        minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
        minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
        minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]
        

    except(Exception):

        x_data=np.array(original_data.loc[:,'Stim_amp_pA'])
        y_data=np.array(original_data.loc[:,'Frequency_Hz'])
        
        maximum_fit_frequency_index = np.nanargmax(y_data)
        maximum_fit_stimulus = x_data[maximum_fit_frequency_index]
        noisy_spike_freq_threshold = np.nanmax([.04*np.nanmax(y_data),2.])
        #noisy_spike_freq_threshold = 2.
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        

        
        minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
        minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
        minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]
        

        
    
    finally:
        
        if do_plot:
            minimum_freq_step_plot = ggplot(original_data_table,aes(x='Stim_amp_pA',y='Frequency_Hz'))+geom_point()
            minimum_freq_step_plot += geom_point(until_maximum_data,aes(x='Stim_amp_pA',y='Frequency_Hz'),color='red')
            #minimum_freq_step_plot += geom_line(full_sigmoid_fit_table, aes(x='Stim_amp_pA',y="Frequency_Hz"))
            minimum_freq_step_plot += geom_vline(xintercept = minimum_freq_step_stim )
            minimum_freq_step_plot += geom_hline(yintercept = minimum_freq_step)
            minimum_freq_step_plot += geom_hline(yintercept = noisy_spike_freq_threshold,linetype='dashed',alpha=.4)
            
            print(minimum_freq_step_plot)
            
        return minimum_freq_step,minimum_freq_step_stim,noisy_spike_freq_threshold,np.nanmax(original_data_table['Frequency_Hz'])


def get_frequency_array(full_cell_id_list):
    
    frequency_table=pd.DataFrame(columns=['Cell_id','Frequency_Array','Obs'])
    saving_folder_path="/Volumes/Work_Julien/Cell_Data_File/"
    original_files = [f for f in glob.glob(str(saving_folder_path+'*.h5*'))]
    no_file_id=[]
    
    for cell_id in tqdm(full_cell_id_list):
        cell_file_path = str(saving_folder_path+'Cell_' +
                         str(cell_id)+'_data_file.h5')
        
        if cell_file_path in original_files:
                
            
            Full_TPC_table,full_SF_dict_table,full_SF_table,Metadata_table,sweep_info_table,Sweep_QC_table,cell_fit_table,cell_feature_table=import_h5_file(cell_file_path,['All'])
            stim_freq_table = fitlib.get_stim_freq_table(full_SF_table.copy(), sweep_info_table.copy(), .5)
            try:
                frequency_array = get_min_freq_step(stim_freq_table,do_plot=False)
                new_line=pd.Series([str(cell_id),frequency_array,"OK"],index=['Cell_id','Frequency_Array','Obs'])
                frequency_table=frequency_table.append(new_line,ignore_index=True)
            except Exception as e:
                new_line=pd.Series([str(cell_id),np.array(stim_freq_table['Frequency_Hz']),str(e)],index=['Cell_id','Frequency_Array','Obs'])
                frequency_table=frequency_table.append(new_line,ignore_index=True)
        else:
            no_file_id.append(cell_id)
    return frequency_table,no_file_id

   
    
def summarize_database(cell_id_list):
    feature_5ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_10ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_25ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_50ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_100ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_250ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    feature_500ms_table=pd.DataFrame(columns=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    unit_line=pd.Series(['--','--','Hz/pA','pA','Hz','pA','Spike_index'],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
    
    feature_5ms_table = feature_5ms_table.append(unit_line,ignore_index=True)
    feature_10ms_table = feature_10ms_table.append(unit_line,ignore_index=True)
    feature_25ms_table = feature_25ms_table.append(unit_line,ignore_index=True)
    feature_50ms_table = feature_50ms_table.append(unit_line,ignore_index=True)
    feature_100ms_table = feature_100ms_table.append(unit_line,ignore_index=True)
    feature_250ms_table = feature_250ms_table.append(unit_line,ignore_index=True)
    feature_500ms_table = feature_500ms_table.append(unit_line,ignore_index=True)
    
    
    cell_linear_values = pd.DataFrame(columns=['Cell_id','Input_Resistance_MOhms','Input_Resistance_MOhms_SD','Time_constant_ms','Time_constant_ms_SD'])
    linear_values_unit = pd.Series(['--','MOhms','MOhms','ms','ms'],index=['Cell_id','Input_Resistance_MOhms','Input_Resistance_MOhms_SD','Time_constant_ms','Time_constant_ms_SD'])
    cell_linear_values = cell_linear_values.append(linear_values_unit,ignore_index=True)
    
    for cell_id in tqdm(cell_id_list):
        Full_TPC_table,full_SF_dict_table,full_SF_table,Metadata_table,sweep_info_table,Sweep_QC_table,cell_fit_table,cell_feature_table=import_h5_file(str('/Volumes/Work_Julien/Cell_Data_File/Cell_'+str(cell_id)+'_data_file.h5'),['Sweep_info','Cell_Feature_Fit'])
        
        IR_mean = np.nanmean(sweep_info_table['Input_Resistance_MOhm'])
        IR_SD = np.nanstd(sweep_info_table['Input_Resistance_MOhm'])
        
        Time_cst_mean = np.nanmean(sweep_info_table['Time_constant_ms'])
        Time_cst_SD = np.nanstd(sweep_info_table['Time_constant_ms'])
        
        cell_linear_values_line = pd.Series([str(cell_id),IR_mean,IR_SD,Time_cst_mean,Time_cst_SD],index = ['Cell_id','Input_Resistance_MOhms','Input_Resistance_MOhms_SD','Time_constant_ms','Time_constant_ms_SD'])
        cell_linear_values=cell_linear_values.append(cell_linear_values_line,ignore_index=True)
        
        

        
        if 5 not in cell_feature_table['Response_time_ms']:
            line_5ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[5,'I_O_obs']
            Gain = cell_feature_table.loc[5,"Gain"]
            Threshold = cell_feature_table.loc[5,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[5,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[5,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[5,"Adaptation_index"]
            
            line_5ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_5ms_table = feature_5ms_table.append(line_5ms,ignore_index=True)
        
        if 10 not in cell_feature_table['Response_time_ms']:
            line_10ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[10,'I_O_obs']
            Gain = cell_feature_table.loc[10,"Gain"]
            Threshold = cell_feature_table.loc[10,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[10,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[10,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[10,"Adaptation_index"]
            
            line_10ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_10ms_table = feature_10ms_table.append(line_10ms,ignore_index=True)
        
        if 25 not in cell_feature_table['Response_time_ms']:
            line_25ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[25,'I_O_obs']
            Gain = cell_feature_table.loc[25,"Gain"]
            Threshold = cell_feature_table.loc[25,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[25,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[25,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[25,"Adaptation_index"]
            
            line_25ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_25ms_table = feature_25ms_table.append(line_25ms,ignore_index=True)
        
        if 50 not in cell_feature_table['Response_time_ms']:
            line_50ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[50,'I_O_obs']
            Gain = cell_feature_table.loc[50,"Gain"]
            Threshold = cell_feature_table.loc[50,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[50,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[50,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[50,"Adaptation_index"]
            
            line_50ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_50ms_table = feature_50ms_table.append(line_50ms,ignore_index=True)
        
        if 100 not in cell_feature_table['Response_time_ms']:
            line_100ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[100,'I_O_obs']
            Gain = cell_feature_table.loc[100,"Gain"]
            Threshold = cell_feature_table.loc[100,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[100,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[100,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[100,"Adaptation_index"]
            
            line_100ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_100ms_table = feature_100ms_table.append(line_100ms,ignore_index=True)
        
        if 250 not in cell_feature_table['Response_time_ms']:
            line_250ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[250,'I_O_obs']
            Gain = cell_feature_table.loc[250,"Gain"]
            Threshold = cell_feature_table.loc[250,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[250,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[250,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[250,"Adaptation_index"]
            
            line_250ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_250ms_table = feature_250ms_table.append(line_250ms,ignore_index=True)
        
        if 500 not in cell_feature_table['Response_time_ms']:
            line_500ms = pd.Series([str(cell_id),'No_I_O_Adapt_computed',np.nan,np.nan,np.nan,np.nan,np.nan],index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        else:
            I_O_obs = cell_fit_table.loc[500,'I_O_obs']
            Gain = cell_feature_table.loc[500,"Gain"]
            Threshold = cell_feature_table.loc[500,"Threshold"]
            Saturation_Frequency = cell_feature_table.loc[500,"Saturation_Frequency"]
            Saturation_Stimulus = cell_feature_table.loc[500,"Saturation_Stimulus"]
            Adaptation_index = cell_feature_table.loc[500,"Adaptation_index"]
            
            line_500ms = pd.Series([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Adaptation_index],
                                 index=['Cell_id','Obs','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Adaptation_index'])
        
        feature_500ms_table = feature_500ms_table.append(line_500ms,ignore_index=True)
    
    feature_5ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_5ms.csv')
    feature_10ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_10ms.csv')
    feature_25ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_25ms.csv')
    feature_50ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_50ms.csv')
    feature_100ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_100ms.csv')
    feature_250ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_250ms.csv')
    feature_500ms_table.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Feature_Table_500ms.csv')
    
    cell_linear_values.to_csv('/Users/julienballbe/My_Work/Data_Analysis/Full_population_files/Full_Cell_linear_values.csv')
    
    return 'Done'
            


    

#%%




