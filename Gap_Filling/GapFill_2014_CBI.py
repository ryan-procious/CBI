import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def read_wl_csv(file_path):
    wl_df = pd.read_csv(file_path)

    ##### if the csv is from lighthouse then this drop function is always true
    ##### if the csv is not from lighthouse then you will need to modify the function

    wl_df.drop(labels=range(len(wl_df)-6,len(wl_df)), axis=0, inplace=True)

    keys = wl_df.keys().to_list()
    
    wl_df['date'] = pd.to_datetime(wl_df[keys[0]])
    wl_df[keys[1]].replace([-999, -99, 99, 'NA', 'RM'], np.nan, inplace=True)
    wl_df[keys[2]].replace([-999, -99, 99, 'NA', 'RM'], np.nan, inplace=True)
    wl_df[keys[3]].replace([-999, -99, 99, 'NA', 'RM'], np.nan, inplace=True)
    wl_df['pwl'] = pd.to_numeric(wl_df[keys[1]],errors= 'coerce')
    wl_df['bwl'] = pd.to_numeric(wl_df[keys[2]],errors= 'coerce')
    wl_df['harmwl'] = pd.to_numeric(wl_df[keys[3]],errors= 'coerce')
    wl_df['pwl surge'] = wl_df['pwl'] - wl_df['harmwl']
    wl_df['bwl surge'] = wl_df['bwl'] - wl_df['harmwl']
    del keys
    return wl_df

def locate_gaps(WL_data):
    lengthMissVal = []
    dates = []
    count = 0
    for i in range(len(WL_data)):
        if pd.isna(WL_data['pwl surge'][i]):
            if count == 0:  # Start of a new NaN gap
                dates.append(WL_data['date'][i])  # Record the start date of the gap
            count += 1  # Increment the gap length

        else:
            if count > 0:  # End of a NaN gap
                lengthMissVal.append(count)
                count = 0  # Reset count after recording the gap length

    # Finalize the DataFrame
    WL_data_gaps = pd.DataFrame()
    WL_data_gaps['date'] = pd.to_datetime(dates)
    WL_data_gaps['gapLength'] = lengthMissVal
    WL_data_gaps['gapTime(min)'] = WL_data_gaps['gapLength'] * 6

    del lengthMissVal,dates,count

    return WL_data_gaps

def eligible_gap_length(WL_gaps): #Function to sort the lengh of the gaps into three categories
    WL_gaps_filter_6min = WL_gaps['gapLength'] == 1
    WL_gaps_filter = (WL_gaps['gapLength'] <= 576) & (WL_gaps['gapLength'] > 1)

    #filters the data into individual dataframes
    linear_gaps = WL_gaps[WL_gaps_filter_6min]
    gaps_less_5_days = WL_gaps[WL_gaps_filter]

    del WL_gaps_filter,WL_gaps_filter_6min

    return linear_gaps,gaps_less_5_days




def linear_fill(Wl_data,linear_gaps): #function to fill in gaps with length of 1 using linear approach

    matching_dates = Wl_data[Wl_data['date'].isin(linear_gaps['date'])]

    index_locations = matching_dates.index.tolist()

    for i in range(len(index_locations)):
        new_value = (Wl_data.loc[(index_locations[i])-1,'pwl surge']+ Wl_data.loc[index_locations[i]+1,'pwl surge']) / 2
        Wl_data.loc[index_locations[i],'pwl surge'] = new_value

    del matching_dates, index_locations, new_value
    
    return Wl_data


def check_bwl(Wl_data,gaps):

    matching_dates = Wl_data[Wl_data['date'].isin(gaps['date'])]

    index_locations = matching_dates.index.tolist()

    gap_length = gaps['gapLength'].tolist()

    valid_gaps = []

    for i in range(len(index_locations)):

        is_valid = Wl_data['bwl surge'][index_locations[i]:index_locations[i]+gap_length[i]].isna().sum() == 0
        valid_gaps.append(is_valid)
    
    filtered_gaps = gaps[valid_gaps].reset_index(drop=True)

    del matching_dates, index_locations, gap_length, valid_gaps, is_valid

    return filtered_gaps
         



def poly_gap_fill(Wl_data, gaps):

    poly_df_list = list()
    
    matching_dates = Wl_data[Wl_data['date'].isin(gaps['date'])]

    index_locations = matching_dates.index.tolist()

    gap_length = gaps['gapLength'].tolist()

    gap_date_list = list()

    for i in range(len(matching_dates)):

        gap_date_df = pd.DataFrame()

        gap_date_df['date'] = Wl_data['date'][index_locations[i]:index_locations[i]+gap_length[i]]

        gap_date_list.append(gap_date_df)
        

    for i in range(len(index_locations)):

        if index_locations[i]- 2161  > 0 and index_locations[i]+2161+gap_length[i] < len(Wl_data):


            pwl_30_days = Wl_data['pwl surge'][(index_locations[i]- 2160):index_locations[i]+2160+gap_length[i]].tolist()

            bwl_30_days = Wl_data['bwl surge'][(index_locations[i]- 2160):index_locations[i]+2160+gap_length[i]].tolist()

            dates = Wl_data['date'][(index_locations[i]- 2160):index_locations[i]+2160+gap_length[i]].tolist()

            bwl_30_days_with_constants = sm.add_constant(bwl_30_days)

            model = sm.OLS(pwl_30_days,bwl_30_days_with_constants,missing='drop')

            results = model.fit()

            slope = results.params[1]

            intercept = results.params[0]

            
            poly_df = pd.DataFrame({'bwl surge': bwl_30_days, 'pwl surge': pwl_30_days,'date' : pd.to_datetime(dates)})

            poly_df['mwl surge'] = intercept + slope*poly_df['bwl surge']

            poly_df.loc[abs(poly_df['mwl surge'] - poly_df['pwl surge']) > 0.1, ['mwl surge', 'pwl surge']] = np.nan

            
            if poly_df['bwl surge'].isna().sum() + poly_df['pwl surge'].isna().sum() < len(poly_df)*0.1:


                poly_df_copy = poly_df.copy()

                poly_df_copy.dropna(inplace=True)

                poly =np.polynomial.polynomial.Polynomial.fit(poly_df_copy['pwl surge'],poly_df_copy['bwl surge'],4)

                pred_values = poly(poly_df['bwl surge'].values)

                poly_df['mwl surge'] = pred_values

                poly_df_list.append(poly_df)

                del poly_df_copy, poly, pred_values

    matched_dates1 = []
    matched_dates2 = []

    for df1, df2 in zip(gap_date_list, poly_df_list):

        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        
       
        common_dates = df1['date'][df1['date'].isin(df2['date'])]
        
        filtered_df1 = df1[df1['date'].isin(common_dates)]
        filtered_df2 = df2[df2['date'].isin(common_dates)]
        
        
        matched_dates1.append(filtered_df1)
        matched_dates2.append(filtered_df2)

    match_df_1 = pd.concat(matched_dates1, ignore_index=True)
    match_df_2 = pd.concat(matched_dates2, ignore_index=True)

    


    Wl_data_total = match_df_2.merge(Wl_data,on='date', how='outer')

    Wl_data_total = Wl_data_total.drop(columns='bwl surge_x',axis=0)
    Wl_data_total = Wl_data_total.drop(columns='pwl surge_x',axis=0)
    Wl_data_total = Wl_data_total.drop(columns='#date+time',axis=0)

    Wl_data_total['pwl surge'] = Wl_data_total['pwl surge_y']
    Wl_data_total['bwl surge'] = Wl_data_total['bwl surge_y']

    Wl_data_total = Wl_data_total.drop(columns='pwl surge_y',axis=0)
    Wl_data_total = Wl_data_total.drop(columns='bwl surge_y',axis=0)


    del poly_df_list, matching_dates, index_locations, gap_length, gap_date_list, matched_dates1, matched_dates2, match_df_1, match_df_2


    return Wl_data_total



def cbi_gapfill(filepath):

    wl_dataset = read_wl_csv(filepath)

    print('Reading dataset')

    Wl_gaps = locate_gaps(wl_dataset)

    print('Total number of gaps: ', len(Wl_gaps))

    linear_gaps,multi_gaps = eligible_gap_length(Wl_gaps)

    print('Number of Linear Gaps filled:', len(linear_gaps))

    dataset_LF = linear_fill(wl_dataset,linear_gaps)

    print('Single gaps filled')

    valid_multi_gaps = check_bwl(dataset_LF,multi_gaps)

    print('Number of gaps with backup water level:', len(valid_multi_gaps))


    filled_wl_dataset = poly_gap_fill(dataset_LF,valid_multi_gaps)

    print('Gaps filled')

    return filled_wl_dataset , wl_dataset
    
