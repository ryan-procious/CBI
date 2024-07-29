import pandas as pd
import os

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
