import pandas as pd
import os
import numpy as np
def Read_Decoding_Data(file_path):
    df1 = pd.read_csv(file_path, header=None)
    df1 = df1.rename(columns={
        0: 'Station #', 
        1: 'Date', 
        2: 'M1', 
        3: 'M2', 
        4: 'M3', 
        5: 'M4', 
        6: 'M5', 
        7: 'M6', 
        8: 'M7', 
        9: 'M8', 
        10: 'M9', 
        11: 'M10'
    })
    df1 = df1.drop(columns=['Station #', 'Date'])
    transposed_df = df1.T
    long_column_df = transposed_df.unstack().reset_index(drop=True)
    start_date = pd.to_datetime("01/01/1996 00:00")
    date_range = pd.date_range(start=start_date, periods=len(long_column_df), freq='6T')
    final_df = pd.DataFrame({
        'Date': date_range,
        'Value': long_column_df
    })
    return final_df


def process_all_files_in_folder(folder_path):
    dataframes_dict = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            processed_df = Read_Decoding_Data(file_path)
            dataframes_dict[filename] = processed_df
            print(f"Processed {filename}")
    return dataframes_dict


def clean_data(station):
    station.replace([-999, -99, 99, 'NA', 'RM','NaN'], np.nan, inplace=True)
    station['Date'] = pd.to_datetime(station['Date'])
    station['NES'] = pd.to_numeric(station['NES'], errors='coerce')
    station['NWS'] = pd.to_numeric(station['NWS'], errors='coerce')
    station['VLG'] = pd.to_numeric(station['VLG'], errors='coerce')
    station['XPT'] = pd.to_numeric(station['XPT'], errors='coerce')
    station['XMD'] = pd.to_numeric(station['XMD'], errors='coerce')
    return station


def avaliable_points(statoin):
    summary = pd.DataFrame(index=['Count'])
    NES_count = statoin['NES'].isna().sum() - len(statoin)
    NWS_count = statoin['NWS'].isna().sum() - len(statoin)
    VLG_count = statoin['VLG'].isna().sum() - len(statoin)
    XMD_count = statoin['XMD'].isna().sum() - len(statoin)
    XPT_count = statoin['XPT'].isna().sum() - len(statoin)
    summary['NES'] = [abs(NES_count)]
    summary['NWS'] = [abs(NWS_count)]
    summary['VLG'] = [abs(VLG_count)]
    summary['XMD'] = [abs(XMD_count)]
    summary['XPT'] = [abs(XPT_count)]
    return summary

def date_index_locater(start_date,end_date,comparison_df):
    indices = []
# Filter rows by date range
    date_range_filter = (comparison_df['Dates'] >= start_date) & (comparison_df['Dates'] <= end_date)

# Get the indices of the rows within the date range
    indices = (comparison_df[date_range_filter].index.tolist())

    return(indices)