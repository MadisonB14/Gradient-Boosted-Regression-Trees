import numpy as np # type: ignore
import h5py # type: ignore
import warnings
import pandas as pd # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

#path where script and data should be run from
#os.chdir('OneDrive - USDA/Documents/Research/ASF_Project/AllSimulationData')

#####################################################################
#### Structure of 1st set of mat files by column with definition ####
#################### Sensitivity_1 ##################################
# Births - new births over whole grid
# Cn - # of infectious carcasses over whole grid
# En - # of exposed over whole grid
# In - # of infectious over whole grid
# Incidence - new cases over whole grid
# Rn - # of recovered/alive over whole grid
# Sn - # of susceptible over whole grid
# Zn - # of uninfectious carcasses over whole grid
#####################################################################

rep1A = h5py.File('../Sensitivity1_1to1154.mat','r')     
Incid_1A = rep1A.get('Incidence')
Incid_1A = np.delete(Incid_1A, range(1154,14400), axis=2)
# print("Shape of rep1A:", Incid_1A.shape)

rep1B = h5py.File('../Sensitivity1_1155to1355.mat','r')     
Incid_1B = rep1B.get('Incidence')
Incid_1B = np.delete(Incid_1B, range(201,7845), axis=2)
# print("Shape of rep1B:", Incid_1B.shape)
# print(Incid_1B[:, :, 7000])

rep1C = h5py.File('../Sensitivity1_1356to2280.mat','r')     
Incid_1C = rep1C.get('Incidence')
Incid_1C = np.delete(Incid_1C, range(925,7644), axis=2)
# print("Shape of rep1C:", Incid_1C.shape)

rep1DA = h5py.File('../Sensitivity1_2281and4081.mat','r')     
Incid_1DA = rep1DA.get('Incidence')
Incid_1DA = np.delete(Incid_1DA, 1, axis=2)
# print("Shape of rep1DA:", Incid_1DA.shape)

rep1DB = h5py.File('../Sensitivity1_2282to4080.mat','r')     
Incid_1DB = rep1DB.get('Incidence')
Incid_1DB = np.delete(Incid_1DB, range(1799,6718), axis=2)
# print("Shape of rep1DB:", Incid_1DB.shape)

rep1DC = h5py.File('../Sensitivity1_2281and4081.mat','r')     
Incid_1DC = rep1DC.get('Incidence')
Incid_1DC = np.delete(Incid_1DC, 0, axis=2)
# print("Shape of rep1DC:", Incid_1DC.shape)

rep1E = h5py.File('../Sensitivity1_4082to5293.mat','r')     
Incid_1E = rep1E.get('Incidence')
Incid_1E= np.delete(Incid_1E, range(1212,4918), axis=2)
# print("Shape of rep1E:", Incid_1E.shape)

rep1F = h5py.File('../Sensitivity1_5294to6241.mat','r')     
Incid_1F = rep1F.get('Incidence')
Incid_1F = np.delete(Incid_1F, range(948,3706), axis=2)
# print("Shape of rep1F:", Incid_1F.shape)

rep1G = h5py.File('../Sensitivity1_6242to6961.mat','r')     
Incid_1G = rep1G.get('Incidence')
Incid_1G = np.delete(Incid_1G, range(720,2758), axis=2)
# print("Shape of rep1G:", Incid_1G.shape)

rep1H = h5py.File('../Sensitivity1_6962to8281.mat','r')     
Incid_1H = rep1H.get('Incidence')
Incid_1H = np.delete(Incid_1H, range(1320,2038), axis=2)
# print("Shape of rep1H:", Incid_1H.shape)

rep1I = h5py.File('../Sensitivity1_8282to8999.mat','r')     
Incid_1I = rep1I.get('Incidence')
# print("Shape of rep1I:", Incid_1I.shape)

rep1J = h5py.File('../Sensitivity1_9000to9889.mat','r')     
Incid_1J = rep1J.get('Incidence')
Incid_1J = np.delete(Incid_1J, range(890,3000), axis=2)
# print("Shape of rep1J:", Incid_1J.shape)

rep1K = h5py.File('../Sensitivity1_9890to10561.mat','r')     
Incid_1K = rep1K.get('Incidence')
Incid_1K = np.delete(Incid_1K, range(672,2110), axis=2)
# print("Shape of rep1K:", Incid_1K.shape)

rep1L = h5py.File('../Sensitivity1_10562to11065.mat','r')     
Incid_1L = rep1L.get('Incidence')
Incid_1L = np.delete(Incid_1L, range(504,1438), axis=2)
# print("Shape of rep1L:", Incid_1L.shape)

rep1M = h5py.File('../Sensitivity1_11066to11380.mat','r')     
Incid_1M = rep1M.get('Incidence')
Incid_1M = np.delete(Incid_1M, range(315,934), axis=2)
# print("Shape of rep1M:", Incid_1M.shape)

rep1N = h5py.File('../Sensitivity1_11381to11499.mat','r')     
Incid_1N = rep1N.get('Incidence')
# print("Shape of rep1N:", Incid_1N.shape)

rep1O = h5py.File('../Sensitivity1_11500to11999.mat','r')     
Incid_1O = rep1O.get('Incidence')
# print("Shape of rep1O:", Incid_1O.shape)

rep1P = h5py.File('../Sensitivity1_12000to13681.mat','r')     
Incid_1P = rep1P.get('Incidence')
Incid_1P = np.delete(Incid_1P, range(1682,2401), axis=2)
# print("Shape of rep1P:", Incid_1P.shape)

rep1Q = h5py.File('../Sensitivity1_13682to14400.mat','r')     
Incid_1Q = rep1Q.get('Incidence')
# print("Final z dimension:", Incid_1Q[0, 1, :])
# print("Shape of rep1Q:", Incid_1Q.shape)

file_list = [Incid_1A, Incid_1B, Incid_1C, Incid_1DA,
             Incid_1DB, Incid_1DC, Incid_1E, Incid_1F,
             Incid_1G, Incid_1H, Incid_1I, Incid_1J,
             Incid_1K, Incid_1L, Incid_1M, Incid_1N, 
             Incid_1O, Incid_1P, Incid_1Q]

model1 = np.empty([100, 72, 0])

for file in file_list:
    file = np.array(file)
    model1=np.dstack((model1, file))

num_reps = model1.shape[0]
num_weeks = model1.shape[1]
num_systems = model1.shape[2]
peak_cases = np.zeros((num_reps, num_systems))
peak_week = np.zeros((num_reps, num_systems))

# Loop through each system
for system_idx in range(num_systems):  
    # Loop through each rep
    for rep_idx in range(num_reps):
        # Sum values across all weeks for the current system and rep
        weekly_sum = np.sum(model1[rep_idx, :, system_idx])
        
        # If the sum is greater than 10, there is an outbreak (<= 10 means no outbreak)
        if weekly_sum > 10:
            # Find the week with the highest value for this rep and system
            week_max = np.argmax(model1[rep_idx, :, system_idx])
            peak_value = model1[rep_idx, week_max, system_idx]
            
            # Only update peak_cases and peak_week if the peak value is greater than 1
            # This is checking for long term stuttering
            if peak_value > 1: 
                peak_cases[rep_idx, system_idx] = peak_value
                peak_week[rep_idx, system_idx] = week_max

# final shape of peak_cases would be 14400 x 100
# flatten this and transpose
# final shape (1440000,)
flat_peak_cases = peak_cases.T.flatten()
flat_peak_cases = pd.DataFrame(flat_peak_cases)
# final shape of peak_week would be 14400 x 100
# flatten this and transpose
# final shape (1440000,)
flat_peak_week = peak_week.T.flatten()
flat_peak_week = pd.DataFrame(flat_peak_week)

# Correctly repeats the parameters to match up with the flattened output data
parameters_df = pd.read_csv('../FullSetOfParameters.csv')  # Shape: 14400 x 7
parameters_repeated = pd.DataFrame(np.repeat(parameters_df.values, 100, axis=0))

data_df = pd.concat([parameters_repeated, flat_peak_cases, flat_peak_week], axis=1, ignore_index=True)
print(data_df.shape)

new_columns = ['Movement', 'HostDensity', 'InfectPeriodLive', 'InfectPeriodCarcass', 
               'IncubationPeriod','ProportionRecover', 'DaysToRecovery', 'PeakCases', 'PeakWeek']
    
data_df.columns = new_columns # had to re-add column names

# Filter out rows where 'PeakCases' is 0
df_filtered = data_df[data_df.iloc[:, 7] !=0]

# Create the 6 dataframes based on combinations of Movement and HostDensity
# Save Output data to file
dfg_lowdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 1.5)]
dfg_lowdens_movement1.iloc[:, 8].to_csv('LowDens-LowMove-PeakWeek.csv', index=False, header=False)
print(dfg_lowdens_movement1.shape)

dfg_meddens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 3)]
dfg_meddens_movement1.iloc[:, 8].to_csv('MedDens-LowMove-PeakWeek.csv', index=False, header=False)
print(dfg_meddens_movement1.shape)

dfg_highdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 5)]
dfg_highdens_movement1.iloc[:, 8].to_csv('HighDens-LowMove-PeakWeek.csv', index=False, header=False)
print(dfg_highdens_movement1.shape)

dfg_lowdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 1.5)]
dfg_lowdens_movement2.iloc[:, 8].to_csv('LowDens-HighMove-PeakWeek.csv', index=False, header=False)
print(dfg_lowdens_movement2.shape)

dfg_meddens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 3)]
dfg_meddens_movement2.iloc[:, 8].to_csv('MedDens-HighMove-PeakWeek.csv', index=False, header=False)
print(dfg_meddens_movement2.shape)

dfg_highdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 5)]
dfg_highdens_movement2.iloc[:, 8].to_csv('HighDens-HighMove-PeakWeek.csv', index=False, header=False)
print(dfg_highdens_movement2.shape)

#### LOG PEAK CASES
df_test = df_filtered[df_filtered['PeakCases'] > 0].copy()
df_test['log10PeakCases'] = np.log10(df_filtered['PeakCases'])
# df_test.iloc[:, 9].to_csv('log10PeakCases.csv', index=False, header=False)

# Create the 6 dataframes based on combinations of Movement and HostDensity
# Save Output data to file
df_lowdens_movement1 = df_test[(df_test['Movement'] == 1) & (df_test['HostDensity'] == 1.5)]
df_lowdens_movement1.iloc[:, 9].to_csv('LowDens-LowMove-log10PeakCases.csv', index=False, header=False)
print(df_lowdens_movement1.shape)

df_meddens_movement1 = df_test[(df_test['Movement'] == 1) & (df_test['HostDensity'] == 3)]
df_meddens_movement1.iloc[:, 9].to_csv('MedDens-LowMove-log10PeakCases.csv', index=False, header=False)
print(df_meddens_movement1.shape)

df_highdens_movement1 = df_test[(df_test['Movement'] == 1) & (df_test['HostDensity'] == 5)]
df_highdens_movement1.iloc[:, 9].to_csv('HighDens-LowMove-log10PeakCases.csv', index=False, header=False)
print(df_highdens_movement1.shape)

df_lowdens_movement2 = df_test[(df_test['Movement'] == 2) & (df_test['HostDensity'] == 1.5)]
df_lowdens_movement2.iloc[:, 9].to_csv('LowDens-HighMove-log10PeakCases.csv', index=False, header=False)
print(df_lowdens_movement2.shape)

df_meddens_movement2 = df_test[(df_test['Movement'] == 2) & (df_test['HostDensity'] == 3)]
df_meddens_movement2.iloc[:, 9].to_csv('MedDens-HighMove-log10PeakCases.csv', index=False, header=False)
print(df_meddens_movement2.shape)

df_highdens_movement2 = df_test[(df_test['Movement'] == 2) & (df_test['HostDensity'] == 5)]
df_highdens_movement2.iloc[:, 9].to_csv('HighDens-HighMove-log10PeakCases.csv', index=False, header=False)
print(df_highdens_movement2.shape)


