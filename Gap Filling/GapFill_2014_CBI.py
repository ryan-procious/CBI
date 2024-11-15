import pandas as pd
import numpy.polynomial
from scipy.stats import linregress
import numpy as np
import os
import matplotlib.pyplot as plt

def locate_gaps(WL_data):
    lengthMissVal = []
    dates = []
    count = 0
    for i in range(len(WL_data)):
        if pd.isna(WL_data['022-pwl'][i]):
            if count == 0:  # Start of a new NaN gap
                dates.append(WL_data['#date+time'][i])  # Record the start date of the gap
            count += 1  # Increment the gap length

        else:
            if count > 0:  # End of a NaN gap
                lengthMissVal.append(count)
                count = 0  # Reset count after recording the gap length

    # Finalize the DataFrame
    WL_data_gaps = pd.DataFrame()
    WL_data_gaps['date'] = dates
    WL_data_gaps['gapLength'] = lengthMissVal
    WL_data_gaps['gapTime(min)'] = WL_data_gaps['gapLength'] * 6
    return WL_data_gaps


def eligible_gap_length(WL_gaps): #Function to sort the lengh of the gaps into three categories
    WL_gaps_filter_6min = WL_gaps['gapLength'] == 1
    WL_gaps_filter_3hr = WL_gaps['gapLength'] <= 24
    WL_gaps_filter_3days = WL_gaps['gapLength'] <= 432

    #filters the data into individual dataframes
    linear_gaps = WL_gaps[WL_gaps_filter_6min]
    three_hr_gaps = WL_gaps[WL_gaps_filter_3hr]
    three_day_gaps = WL_gaps[WL_gaps_filter_3days]

    return linear_gaps,three_hr_gaps,three_day_gaps




def linear_fill(Wl_data,linear_gaps): #function to fill in gaps with length of 1 using linear approach

    matching_dates = Wl_data[Wl_data['date'].isin(linear_gaps['date'])]

    index_locations = matching_dates.index.tolist()

    for i in range(len(index_locations)):
        new_value = (Wl_data.loc[(index_locations[i])-1,'022-pwl']+ Wl_data.loc[index_locations[i]+1,'022-pwl']) / 2
        Wl_data.loc[index_locations[i],'022-pwl'] = new_value
    
    return Wl_data




def poly_gap_fill(Wl_data, three_hr_gaps):
    
    matching_dates = Wl_data[Wl_data['date'].isin(three_hr_gaps['date'])]

    index_locations = matching_dates.index.tolist()

    gap_length = three_hr_gaps['gapLength'].tolist()

    for i in range(len(index_locations)):

        if index_locations[i]- 2160 - gap_length[i] > 0 and index_locations[i]+2160+gap_length[i]:

            pwl_30_days = Wl_data['022-pwl'][(index_locations[i]- 2160 - gap_length[i]):index_locations[i]+2160+gap_length[i]].tolist()

            bwl_30_days = Wl_data['022-bwl'][(index_locations[i]- 2160):index_locations[i]+2160].tolist()
    
        slope, intercept, r, p, se = linregress(pwl_30_days,bwl_30_days)

        poly_df = pd.DataFrame()

        poly_df['bwl'] = bwl_30_days
        poly_df['pwl'] = pwl_30_days
        poly_df['mwl'] = intercept + slope*poly_df['bwl']

        poly_df = poly_df[abs(poly_df['mwl'] - poly_df['pwl']) <= 0.1]
        
        if poly_df['bwl'].isna().sum() + poly_df['pwl'].isna().sum() < len(poly_df)*0.1:

            poly = np.polynomial.polynomial.Polynomial.fit(pwl_30_days,bwl_30_days,4)

            pred_values = poly(poly_df['bwl'].values)

        poly_df['mwl'] = pred_values

        '''avg_5_before = np.mean(Wl_data[index_locations[i]- 6: index_locations[i]-1]['bwl']) - np.mean(Wl_data[index_locations[i]- 6: index_locations[i]-1]['pwl'])

        avg_5_after = np.mean(Wl_data[index_locations[i]+ 1+ gap_length[i]: index_locations[i]+6+ gap_length[i]]['bwl']) - np.mean(Wl_data[index_locations[i]+ 1+ gap_length[i]: index_locations[i]+6+ gap_length[i]]['pwl'])

        poly_df['mwl_adj'] = poly_df['mwl'] + (avg_5_before + )'''

        return poly_df
