# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:19:06 2024

@author: Admin
"""

import h5py
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime, timedelta



def extract_track_info(fname):
    """
    Extract orbit number, date, and cycle information from the file name.

    Parameters:
    fname (str): The file name.

    Returns:
    info (dict): A dictionary containing orbit number, date, and cycle information.
    """
    base_name = os.path.basename(fname)
    parts = base_name.split('_')
    info = {
        'date': parts[1],
        'track': int(parts[2][:4]),
        'cycle': int(parts[2][4:6])
    }
    return info

def gps_time_to_utc(gps_seconds, gps_epoch):
    """
    Convert GPS time to UTC time.

    Parameters:
    gps_seconds (numpy array): GPS time in seconds.
    gps_epoch (datetime): GPS epoch time.

    Returns:
    utc_times (list): A list of UTC times.
    """
    utc_times = []
    for sec in gps_seconds:
        elapsed = timedelta(seconds=sec)
        utc_time = gps_epoch + elapsed
        utc_times.append(utc_time)
    return utc_times

def read_atl13(fname, calculate_time=False, apply_filter=False, extent=None):
    """
   Read ICESat-2 ATL13 data and return specified variable information, including start and end UTC times.
   Optionally apply filters.

   Parameters:
   fname (str): The path to the HDF5 file.
   calculate_time (bool): Whether to calculate time based on delta_time.
   apply_filter (bool): Whether to apply any filter to the data.
   extent (tuple): A tuple defining the filtering range (xmin, xmax, ymin, ymax).

   Returns:
   A dictionary containing all beam data and start/end time, or filtered data and statistics.
   """
    groups = ['/gt1l', '/gt1r', '/gt2l', '/gt2r', '/gt3l', '/gt3r']
    trajectory_data = {}
    gps_epoch = datetime(2018, 1, 1, 0, 0, 0)
    
    with h5py.File(fname, 'r') as fi:
        
        data_start_utc = fi['/ancillary_data/data_start_utc'][0].decode('utf-8')
        data_end_utc = fi['/ancillary_data/data_end_utc'][0].decode('utf-8')

        for g in groups:
          
            delta_time = fi[g + '/delta_time'][:]  
            lon = fi[g + '/segment_lon'][:]  
            lat = fi[g + '/segment_lat'][:]  
            ht_water_surf = fi[g + '/ht_water_surf'][:] 
            segment_geoid = fi[g + '/segment_geoid'][:]  
            ht_ortho = fi[g + '/ht_ortho'][:]  
            segment_dem_ht = fi[g + '/segment_dem_ht'][:] 
            qf_bckgrd = fi[g + '/qf_bckgrd'][:]  
            qf_bias_em = fi[g + '/qf_bias_em'][:]  
            qf_bias_fit = fi[g + '/qf_bias_fit'][:]  
            stdev_water_surf = fi[g + '/stdev_water_surf'][:] 
            significant_wave_ht = fi[g + '/significant_wave_ht'][:] 
           
            if calculate_time:
                utc_time = gps_time_to_utc(delta_time, gps_epoch)
                time_col = pd.Series(utc_time, name='time')
            else:
                time_col = pd.Series([None] * len(lat), name='time')

            result = pd.DataFrame({
                'lon': lon,  
                'lat': lat,  
                'ht_water_surf': ht_water_surf, 
                'segment_geoid': segment_geoid,
                'ht_ortho': ht_ortho,  
                'segment_dem_ht' : segment_dem_ht,  
                'qf_bckgrd': qf_bckgrd, 
                'qf_bias_em': qf_bias_em,  
                'qf_bias_fit': qf_bias_fit,  
                'stdev_water_surf': stdev_water_surf,  
                'significant_wave_ht': significant_wave_ht 
            })


            result = pd.concat([result, time_col], axis=1)
            trajectory_data[g] = result

  
    trajectory_data['data_start_utc'] = data_start_utc
    trajectory_data['data_end_utc'] = data_end_utc
    

    if apply_filter:
       filtered_data, beam_stats, total_stats = filter_atl13_data(trajectory_data, extent)
       return filtered_data, beam_stats, total_stats
     
    return trajectory_data


def filter_atl13_data(data, extent=None):
    """
      Filter ICESat-2 ATL13 data based on different conditions, count the filtered points for each beam, and summarize the results.

     Parameters:
     data (dict): A dictionary containing ATL13 data for multiple beams.
     extent (tuple): A tuple defining the filtering range (xmin, xmax, ymin, ymax) for spatial filtering. If not provided, defaults to global extent.

     Returns:
     dict: A dictionary containing the filtered data, as well as statistical information for each beam and summary statistics.
    """


   
    if extent is None:
        extent = (-180, 180, -90, 90)  
    
    filtered_data = {}
    total_stats = {
        'total_points': 0,
        'filtered_points': 0,
        'removed_points': 0,
        'nan_points': 0, 
        'extent_filtered': 0,  
        'invalid_value_filtered': 0, 
        'qf_bckgrd_filtered': 0, 
        'qf_bias_em_filtered': 0
     
    }

    beam_stats = {}  

    for beam, df in data.items():
        if isinstance(df, pd.DataFrame):
       
            stats = {
                'total_points': 0,
                'filtered_points': 0,
                'removed_points': 0,
                'nan_points': 0,  
                'extent_filtered': 0, 
                'invalid_value_filtered': 0, 
                'qf_bckgrd_filtered': 0, 
                'qf_bias_em_filtered': 0
               
            }

       
            original_point_count = len(df)
            stats['total_points'] = original_point_count
            total_stats['total_points'] += original_point_count


          
            xmin, xmax, ymin, ymax = extent
            valid_idx = (df['lon'] >= xmin) & (df['lon'] <= xmax) & (df['lat'] >= ymin) & (df['lat'] <= ymax)
            extent_filtered_count = original_point_count - valid_idx.sum()
            stats['extent_filtered'] = extent_filtered_count
            total_stats['extent_filtered'] += extent_filtered_count

            df = df.loc[valid_idx]

      
            invalid_value_mask = (df['lat'] == 3.4028235e+38) | (df['lon'] == 3.4028235e+38) | \
                                 (df['ht_water_surf'] == 3.4028235e+38) | (df['stdev_water_surf'] == 3.4028235e+38) | \
                                 (df['qf_bckgrd'] == 3.4028235e+38) | (df['qf_bias_em'] == 3.4028235e+38) | \
                                 (df['qf_bias_fit'] == 3.4028235e+38)| (df['significant_wave_ht'] == 3.4028235e+38)|\
                                 (df['segment_dem_ht'] == 3.4028235e+38)
            invalid_value_filtered_count = invalid_value_mask.sum()
            stats['invalid_value_filtered'] = invalid_value_filtered_count
            total_stats['invalid_value_filtered'] += invalid_value_filtered_count

            df.loc[invalid_value_mask, ['lat', 'lon', 'ht_water_surf', 'segment_dem_ht', 'significant_wave_ht', 'stdev_water_surf', 'qf_bckgrd', 'qf_bias_em', 'qf_bias_fit']] = np.nan

            nan_point_count = df.isna().sum().sum()
            stats['nan_points'] = nan_point_count
            total_stats['nan_points'] += nan_point_count

            df = df.dropna()

          
            qf_bckgrd_valid_idx = df['qf_bckgrd'] < 6
            qf_bckgrd_filtered_count = len(df) - qf_bckgrd_valid_idx.sum()
            stats['qf_bckgrd_filtered'] = qf_bckgrd_filtered_count
            total_stats['qf_bckgrd_filtered'] += qf_bckgrd_filtered_count

            qf_bckgrd_valid_idx_np = qf_bckgrd_valid_idx.to_numpy()

            df = df.loc[qf_bckgrd_valid_idx]

          
            qf_bias_em_valid_idx = (df['qf_bias_em'] >= -2) & (df['qf_bias_em'] <= 2)
            qf_bias_em_filtered_count = len(df) - qf_bias_em_valid_idx.sum()
            stats['qf_bias_em_filtered'] = qf_bias_em_filtered_count
            total_stats['qf_bias_em_filtered'] += qf_bias_em_filtered_count

            qf_bias_em_valid_idx_np = qf_bias_em_valid_idx.to_numpy()

            df = df.loc[qf_bias_em_valid_idx]


          
            filtered_point_count = len(df)
            stats['filtered_points'] = filtered_point_count
            total_stats['filtered_points'] += filtered_point_count
            stats['removed_points'] = original_point_count - filtered_point_count
            total_stats['removed_points'] += stats['removed_points']

            beam_stats[beam] = stats

            filtered_data[beam] = df

    return filtered_data, beam_stats, total_stats


def check_h5_file(file_path):
    """
    Check if the HDF5 file is corrupted.
    
    Parameters:
    file_path (str): Path to the HDF5 file.
    
    Returns:
    bool: Returns True if the file is normal, False if the file is corrupted.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            pass  # Simply attempts to open the file without performing any actual operations
        return True
    except (OSError, IOError):
        return False



def save_data_to_csv(file_path=None, output_dir=None, apply_filter=False, extent=None, calculate_time=False):
    """
    Read ICESat-2 ATL13 data, optionally apply filtering, check for corrupted files and empty beams, and save results to a CSV file.
    
    Parameters:
    file_path (str): Path to the HDF5 file.
    output_dir (str): Directory to save the CSV file. If not provided, it defaults to an 'output' folder in the current directory.
    apply_filter (bool): Whether to apply filtering.
    extent (tuple): A tuple representing the filtering range (xmin, xmax, ymin, ymax).
    calculate_time (bool): Whether to calculate the time.
    
    Returns:
    (list, list): Returns two lists, one containing files with errors and the other containing files with empty beams.
    """

    
    # Initialize lists for corrupted files and files with empty beams
    corrupted_files = []
    empty_beams_files = []

    # Check if the file path exists
    if not os.path.exists(file_path):
        print(f"Warning: The file path {file_path} does not exist. Please check the path.")
        corrupted_files.append(file_path)
        return corrupted_files, empty_beams_files

    # Extract file information
    track_info = extract_track_info(file_path)

    # Check if the file is corrupted
    if not check_h5_file(file_path):
        print(f"The file {file_path} may be corrupted and cannot be opened.")
        corrupted_files.append(file_path)
        return corrupted_files, empty_beams_files

    try:
        # Read the data and apply filtering based on the apply_filter flag
        if apply_filter:
            filtered_data, beam_stats, total_stats = read_atl13(file_path, calculate_time=calculate_time, apply_filter=apply_filter, extent=extent)
            data = filtered_data  # Use the filtered data
        else:
            data = read_atl13(file_path, calculate_time=calculate_time, apply_filter=False, extent=extent)

        # Check if any beam is empty
        all_beams_empty = True
        for beam, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_beams_empty = False
                break  # If at least one beam is not empty, process the file

        # Skip processing if all beams are empty
        if all_beams_empty:
            #print(f"All beam data in the file {file_path} are empty, skipping save.")
            empty_beams_files.append(file_path)
            return corrupted_files, empty_beams_files

        # If no output directory is provided, use the default "output" directory
        if output_dir is None:
            output_dir = 'output'

        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"The output directory {output_dir} does not exist, it has been created.")

    


        # Extract the filename and prepare the output file path
        original_filename = os.path.basename(file_path)
        base_filename = os.path.splitext(original_filename)[0]  # Remove the file extension
        output_file = os.path.join(output_dir, f"{base_filename}.csv")

        # Check if a CSV file with the same name already exists, and overwrite if it does
        if os.path.exists(output_file):
            print(f"The file {output_file} already exists, overwriting.")
        # else:
        #     print(f"Creating file {output_file}.")

        # Open the file and prepare to write
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            for beam, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:  # Check if the dataframe is not empty
                    # Write custom header
                    writer.writerow(["###################"])
                    writer.writerow(['file_name', original_filename])
                    writer.writerow(['beam', beam])  # Write beam and beam name
                    writer.writerow(['track', track_info['track']])  # Write track number
                    writer.writerow(['points', len(df)])  # Number of data points
                    writer.writerow(["###################"])

                    # Write column headers
                    df_columns = df.columns.tolist()
                    writer.writerow(df_columns)

                    # Write data
                    df.to_csv(f, index=False, header=False, mode='a')
                # else:
                #     print(f"No data in beam {beam}.")

        # print(f"Data has been saved to {output_file}")
        return corrupted_files, empty_beams_files  # Only return these two lists

    except OSError as e:
            print(f"Failed to read file {file_path}, it might be corrupted. Error: {e}")
            corrupted_files.append(file_path)
            return corrupted_files, empty_beams_files

    
    
    
if __name__ == "__main__":
        
        # Example usage:
        file_path = 'F:/icesat_atl13/ATL13_20181103005634_05410101_006_01.h5'
        filtered_data, beam_stats, total_stats = read_atl13(file_path, calculate_time=True, apply_filter=True, extent=None)
        save_data_to_csv(file_path, calculate_time=True, apply_filter=True, extent=None)
     
        