import pandas as pd # Use the work with the dataframes
import numpy as np  # Mainly use to assing nan values as np.nan
import matplotlib.pyplot as plt # Visualitzation
from matplotlib.ticker import MaxNLocator # Visualitzation
import statistics  # Uses to compute statistics
import os
import scipy.interpolate
from collections import Counter
import matplotlib.dates as mdates


#function to find the index of a date range
def date_index_locater(start_date,end_date,comparison_df):
    indices = []
# Filter rows by date range
    date_range_filter = (comparison_df['Dates'] >= start_date) & (comparison_df['Dates'] <= end_date)

# Get the indices of the rows within the date range
    indices = (comparison_df[date_range_filter].index.tolist())

    return(indices)


def ten_offset_points(comparison_df):
    differences = comparison_df['Shifted LH WL'] - comparison_df['NOAA WL']
    non_zero_differences = differences != 0

    filtered_differences = differences[non_zero_differences]
    filtered_dates = comparison_df['Dates'][non_zero_differences]

    # Convert the results to DataFrames
    differences_df = filtered_differences.to_frame(name='Difference')
    dates_df = filtered_dates.to_frame(name='Date')

    # Combine the two DataFrames
    remaining_differences = differences_df.join(dates_df, how='inner')

    cleaned_remaining_differences = remaining_differences.dropna()
    cleaned_remaining_differences.reset_index(drop = True, inplace=True)
    mask = abs(cleaned_remaining_differences['Difference']) >= 0.005

    cleaned_remaining_differences = cleaned_remaining_differences[mask]


    for i in range(1, 11):
        cleaned_remaining_differences[f'Next_{i}'] = cleaned_remaining_differences['Difference'].shift(-i)

    # Check if the current value is equal to the next 10 values
    cleaned_remaining_differences['All_Next_10_Equal'] = cleaned_remaining_differences.apply(lambda row: all(row['Difference'] == row[f'Next_{i}'] for i in range(1, 11)), axis=1)

    # Drop the helper columns if they are no longer needed
    cleaned_remaining_differences.drop(columns=[f'Next_{i}' for i in range(1, 11)], inplace=True)


    mask = cleaned_remaining_differences['All_Next_10_Equal'] == True

    # Filter the DataFrame using the mask
    cleaned_remaining_differences = cleaned_remaining_differences[mask]

    # Reset index if needed
    cleaned_remaining_differences.reset_index(drop=True, inplace=True)

    cleaned_remaining_differences.to_clipboard()


def plotting_offsets(start_dates,comparison_df,location):

    for i in range(len(start_dates)):
        plt.plot(comparison_df['Dates'],comparison_df['Shifted LH WL'],label = 'Lighthouse')

        plt.plot(comparison_df['Dates'],comparison_df['NOAA WL'],label = 'NOAA')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=400))

        plt.gcf().autofmt_xdate()

        plt.ylim(1.25,2.25)

        start_date = pd.to_datetime(start_dates[i]) - pd.DateOffset(hours= 1)

        end_date = pd.to_datetime(start_dates[i]) + pd.DateOffset(hours = 4)

        plt.xlim(start_date, end_date)
        plt.legend(frameon = False)
        plt.title(location)
        plt.show()

def find_double_points(comparison_df, dataset):
    # Create a mask to find consecutive duplicate values
    mask = comparison_df[dataset] == comparison_df[dataset].shift(-1)

    # Use the mask to filter the DataFrame and get the double values and dates
    double_values = comparison_df.loc[mask, dataset]
    double_dates = comparison_df.loc[mask, 'Dates']
    
    # Return the count of double dates
    return len(double_dates)