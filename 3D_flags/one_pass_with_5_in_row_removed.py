import pandas as pd
import numpy as np
import os

directory_path = r'/Users/rprocious/Waterlevels_CBI/CBI-2/Corrected_Data_Official/nesscanResult_removedBadNesdisRecords/BobHall'
dataframes = []
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(
            file_path,
            parse_dates=[0],
            na_values=[-999, -99, 99, 'NA', 'RM'],
            engine='python'
        )
        df.columns = ['station #', 'date', 'pwl']
        dataframes.append(df)

bhp = pd.concat(dataframes, ignore_index=True)
bhp['date'] = pd.to_datetime(bhp['date'])
bhp = bhp[bhp['date'] >= pd.to_datetime('1996-01-01')]
bhp = bhp.reset_index(drop=True)

n = len(bhp)
pwl = bhp['pwl'].values
third_diff = np.full(n, np.nan)

# --- First pass to calculate third differences ---
for i in range(3, n):
    window = pwl[i-3:i+1]
    if np.isnan(window).any():
        continue
    third_val = window[0] - 3 * window[1] + 3 * window[2] - window[3]
    third_diff[i] = round(third_val, 5)

# Thresholds using mean ± 6*std
clean_vals = third_diff[~np.isnan(third_diff)]
mean_val = clean_vals.mean()
std_val = clean_vals.std()
threshold_upper = mean_val + 6 * std_val
threshold_lower = mean_val - 6 * std_val

# --- Flagging spikes and repeated values ---
spike_flag = np.zeros(n, dtype=int)
pwl_cleaned = pwl.copy()

i = 3
while i < n:
    if np.isnan(pwl[i-3:i+1]).any():
        i += 1
        continue

    third_val = pwl[i-3] - 3 * pwl[i-2] + 3 * pwl[i-1] - pwl[i]

    if third_val > threshold_upper or third_val < threshold_lower:
        # Flag spike
        spike_flag[i] = 1
        spike_val = pwl[i]
        pwl_cleaned[i] = np.nan

        # Check for repeated values (exact matches) right after the spike
        for j in range(1, 6):  # check next 5 values max
            if i + j >= n:
                break
            if pwl[i + j] == spike_val:
                spike_flag[i + j] = 1
                pwl_cleaned[i + j] = np.nan
            else:
                break
        i += j  # jump past the last repeated value
    else:
        i += 1

# --- Save Results ---
bhp['third_diff'] = third_diff
bhp['spike_flag'] = spike_flag
bhp['pwl_cleaned'] = pwl_cleaned

print(bhp.head(20))
