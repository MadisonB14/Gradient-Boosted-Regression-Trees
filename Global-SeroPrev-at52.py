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
In_1A = rep1A.get('In')
In_1A = np.delete(In_1A, range(1154,14400), axis=2)
Rn_1A = rep1A.get('Rn')
Rn_1A = np.delete(Rn_1A, range(1154,14400), axis=2)
En_1A = rep1A.get('En')
En_1A = np.delete(En_1A, range(1154,14400), axis=2)
Sn_1A = rep1A.get('Sn')
Sn_1A = np.delete(Sn_1A, range(1154,14400), axis=2)
Incid_1A = rep1A.get('Incidence')
Incid_1A = np.delete(Incid_1A, range(1154,14400), axis=2)
# print("Shape of rep1A:", In_1A.shape)

rep1B = h5py.File('../Sensitivity1_1155to1355.mat','r')     
In_1B = rep1B.get('In')
In_1B = np.delete(In_1B, range(201,7845), axis=2)
Rn_1B = rep1B.get('Rn')
Rn_1B = np.delete(Rn_1B, range(201,7845), axis=2)
En_1B = rep1B.get('En')
En_1B = np.delete(En_1B, range(201,7845), axis=2)
Sn_1B = rep1B.get('Sn')
Sn_1B = np.delete(Sn_1B, range(201,7845), axis=2)
Incid_1B = rep1B.get('Incidence')
Incid_1B = np.delete(Incid_1B, range(201,7845), axis=2)
# print("Shape of rep1B:", In_1B.shape)
# print(In_1B[:, :, 7000])

rep1C = h5py.File('../Sensitivity1_1356to2280.mat','r')     
In_1C = rep1C.get('In')
In_1C = np.delete(In_1C, range(925,7644), axis=2)
Rn_1C = rep1C.get('Rn')
Rn_1C = np.delete(Rn_1C, range(925,7644), axis=2)
En_1C = rep1C.get('En')
En_1C = np.delete(En_1C, range(925,7644), axis=2)
Sn_1C = rep1C.get('Sn')
Sn_1C = np.delete(Sn_1C, range(925,7644), axis=2)
Incid_1C = rep1C.get('Incidence')
Incid_1C = np.delete(Incid_1C, range(925,7644), axis=2)
# print("Shape of rep1C:", In_1C.shape)

rep1DA = h5py.File('../Sensitivity1_2281and4081.mat','r')     
In_1DA = rep1DA.get('In')
In_1DA = np.delete(In_1DA, 1, axis=2)
Rn_1DA = rep1DA.get('Rn')
Rn_1DA = np.delete(Rn_1DA, 1, axis=2)
En_1DA = rep1DA.get('En')
En_1DA = np.delete(En_1DA, 1, axis=2)
Sn_1DA = rep1DA.get('Sn')
Sn_1DA = np.delete(Sn_1DA, 1, axis=2)
Incid_1DA = rep1DA.get('Incidence')
Incid_1DA = np.delete(Incid_1DA, 1, axis=2)
# print("Shape of rep1DA:", In_1DA.shape)

rep1DB = h5py.File('../Sensitivity1_2282to4080.mat','r')     
In_1DB = rep1DB.get('In')
In_1DB = np.delete(In_1DB, range(1799,6718), axis=2)
Rn_1DB = rep1DB.get('Rn')
Rn_1DB = np.delete(Rn_1DB, range(1799,6718), axis=2)
En_1DB = rep1DB.get('En')
En_1DB = np.delete(En_1DB, range(1799,6718), axis=2)
Sn_1DB = rep1DB.get('Sn')
Sn_1DB = np.delete(Sn_1DB, range(1799,6718), axis=2)
Incid_1DB = rep1DB.get('Incidence')
Incid_1DB = np.delete(Incid_1DB, range(1799,6718), axis=2)
# print("Shape of rep1DB:", In_1DB.shape)

rep1DC = h5py.File('../Sensitivity1_2281and4081.mat','r')     
In_1DC = rep1DC.get('In')
In_1DC = np.delete(In_1DC, 0, axis=2)
Rn_1DC = rep1DC.get('Rn')
Rn_1DC = np.delete(Rn_1DC, 0, axis=2)
En_1DC = rep1DC.get('En')
En_1DC = np.delete(En_1DC, 0, axis=2)
Sn_1DC = rep1DC.get('Sn')
Sn_1DC = np.delete(Sn_1DC, 0, axis=2)
Incid_1DC = rep1DC.get('Incidence')
Incid_1DC = np.delete(Incid_1DC, 0, axis=2)
# print("Shape of rep1DC:", In_1DC.shape)

rep1E = h5py.File('../Sensitivity1_4082to5293.mat','r')     
In_1E = rep1E.get('In')
In_1E= np.delete(In_1E, range(1212,4918), axis=2)
Rn_1E = rep1E.get('Rn')
Rn_1E= np.delete(Rn_1E, range(1212,4918), axis=2)
En_1E = rep1E.get('En')
En_1E= np.delete(En_1E, range(1212,4918), axis=2)
Sn_1E = rep1E.get('Sn')
Sn_1E= np.delete(Sn_1E, range(1212,4918), axis=2)
Incid_1E = rep1E.get('Incidence')
Incid_1E= np.delete(Incid_1E, range(1212,4918), axis=2)
# print("Shape of rep1E:", In_1E.shape)

rep1F = h5py.File('../Sensitivity1_5294to6241.mat','r')     
In_1F = rep1F.get('In')
In_1F = np.delete(In_1F, range(948,3706), axis=2)
Rn_1F = rep1F.get('Rn')
Rn_1F = np.delete(Rn_1F, range(948,3706), axis=2)
En_1F = rep1F.get('En')
En_1F = np.delete(En_1F, range(948,3706), axis=2)
Sn_1F = rep1F.get('Sn')
Sn_1F = np.delete(Sn_1F, range(948,3706), axis=2)
Incid_1F = rep1F.get('Incidence')
Incid_1F = np.delete(Incid_1F, range(948,3706), axis=2)
# print("Shape of rep1F:", In_1F.shape)

rep1G = h5py.File('../Sensitivity1_6242to6961.mat','r')     
In_1G = rep1G.get('In')
In_1G = np.delete(In_1G, range(720,2758), axis=2)
Rn_1G = rep1G.get('Rn')
Rn_1G = np.delete(Rn_1G, range(720,2758), axis=2)
En_1G = rep1G.get('En')
En_1G = np.delete(En_1G, range(720,2758), axis=2)
Sn_1G = rep1G.get('Sn')
Sn_1G = np.delete(Sn_1G, range(720,2758), axis=2)
Incid_1G = rep1G.get('Incidence')
Incid_1G = np.delete(Incid_1G, range(720,2758), axis=2)
# print("Shape of rep1G:", In_1G.shape)

rep1H = h5py.File('../Sensitivity1_6962to8281.mat','r')     
In_1H = rep1H.get('In')
In_1H = np.delete(In_1H, range(1320,2038), axis=2)
Rn_1H = rep1H.get('Rn')
Rn_1H = np.delete(Rn_1H, range(1320,2038), axis=2)
En_1H = rep1H.get('En')
En_1H = np.delete(En_1H, range(1320,2038), axis=2)
Sn_1H = rep1H.get('Sn')
Sn_1H = np.delete(Sn_1H, range(1320,2038), axis=2)
Incid_1H = rep1H.get('Incidence')
Incid_1H = np.delete(Incid_1H, range(1320,2038), axis=2)
# print("Shape of rep1H:", In_1H.shape)

rep1I = h5py.File('../Sensitivity1_8282to8999.mat','r')     
In_1I = rep1I.get('In')
Rn_1I = rep1I.get('Rn')
En_1I = rep1I.get('En')
Sn_1I = rep1I.get('Sn')
Incid_1I = rep1I.get('Incidence')
# print("Shape of rep1I:", In_1I.shape)

rep1J = h5py.File('../Sensitivity1_9000to9889.mat','r')     
In_1J = rep1J.get('In')
In_1J = np.delete(In_1J, range(890,3000), axis=2)
Rn_1J = rep1J.get('Rn')
Rn_1J = np.delete(Rn_1J, range(890,3000), axis=2)
En_1J = rep1J.get('En')
En_1J = np.delete(En_1J, range(890,3000), axis=2)
Sn_1J = rep1J.get('Sn')
Sn_1J = np.delete(Sn_1J, range(890,3000), axis=2)
Incid_1J = rep1J.get('Incidence')
Incid_1J = np.delete(Incid_1J, range(890,3000), axis=2)
# print("Shape of rep1J:", In_1J.shape)

rep1K = h5py.File('../Sensitivity1_9890to10561.mat','r')     
In_1K = rep1K.get('In')
In_1K = np.delete(In_1K, range(672,2110), axis=2)
Rn_1K = rep1K.get('Rn')
Rn_1K = np.delete(Rn_1K, range(672,2110), axis=2)
En_1K = rep1K.get('En')
En_1K = np.delete(En_1K, range(672,2110), axis=2)
Sn_1K = rep1K.get('Sn')
Sn_1K = np.delete(Sn_1K, range(672,2110), axis=2)
Incid_1K = rep1K.get('Incidence')
Incid_1K = np.delete(Incid_1K, range(672,2110), axis=2)
# print("Shape of rep1K:", In_1K.shape)

rep1L = h5py.File('../Sensitivity1_10562to11065.mat','r')     
In_1L = rep1L.get('In')
In_1L = np.delete(In_1L, range(504,1438), axis=2)
Rn_1L = rep1L.get('Rn')
Rn_1L = np.delete(Rn_1L, range(504,1438), axis=2)
En_1L = rep1L.get('En')
En_1L = np.delete(En_1L, range(504,1438), axis=2)
Sn_1L = rep1L.get('Sn')
Sn_1L = np.delete(Sn_1L, range(504,1438), axis=2)
Incid_1L = rep1L.get('Incidence')
Incid_1L = np.delete(Incid_1L, range(504,1438), axis=2)
# print("Shape of rep1L:", In_1L.shape)

rep1M = h5py.File('../Sensitivity1_11066to11380.mat','r')     
In_1M = rep1M.get('In')
In_1M = np.delete(In_1M, range(315,934), axis=2)
Rn_1M = rep1M.get('Rn')
Rn_1M = np.delete(Rn_1M, range(315,934), axis=2)
En_1M = rep1M.get('En')
En_1M = np.delete(En_1M, range(315,934), axis=2)
Sn_1M = rep1M.get('Sn')
Sn_1M = np.delete(Sn_1M, range(315,934), axis=2)
Incid_1M = rep1M.get('Incidence')
Incid_1M = np.delete(Incid_1M, range(315,934), axis=2)
# print("Shape of rep1M:", In_1M.shape)

rep1N = h5py.File('../Sensitivity1_11381to11499.mat','r')     
In_1N = rep1N.get('In')
Rn_1N = rep1N.get('Rn')
En_1N = rep1N.get('En')
Sn_1N = rep1N.get('Sn')
Incid_1N = rep1N.get('Incidence')
# print("Shape of rep1N:", In_1N.shape)

rep1O = h5py.File('../Sensitivity1_11500to11999.mat','r')     
In_1O = rep1O.get('In')
Rn_1O = rep1O.get('Rn')
En_1O = rep1O.get('En')
Sn_1O = rep1O.get('Sn')
Incid_1O = rep1O.get('Incidence')
# print("Shape of rep1O:", In_1O.shape)

rep1P = h5py.File('../Sensitivity1_12000to13681.mat','r')     
In_1P = rep1P.get('In')
In_1P = np.delete(In_1P, range(1682,2401), axis=2)
Rn_1P = rep1P.get('Rn')
Rn_1P = np.delete(Rn_1P, range(1682,2401), axis=2)
En_1P = rep1P.get('En')
En_1P = np.delete(En_1P, range(1682,2401), axis=2)
Sn_1P = rep1P.get('Sn')
Sn_1P = np.delete(Sn_1P, range(1682,2401), axis=2)
Incid_1P = rep1P.get('Incidence')
Incid_1P = np.delete(Incid_1P, range(1682,2401), axis=2)
# print("Shape of rep1P:", In_1P.shape)

rep1Q = h5py.File('../Sensitivity1_13682to14400.mat','r')     
In_1Q = rep1Q.get('In')
Rn_1Q = rep1Q.get('Rn')
En_1Q = rep1Q.get('En')
Sn_1Q = rep1Q.get('Sn')
Incid_1Q = rep1Q.get('Incidence')

# print("Final z dimension:", In_1Q[0, 1, :])
# print("Shape of rep1Q:", In_1Q.shape)

file_list = [In_1A, In_1B, In_1C, In_1DA,
             In_1DB, In_1DC, In_1E, In_1F,
             In_1G, In_1H, In_1I, In_1J,
             In_1K, In_1L, In_1M, In_1N, 
             In_1O, In_1P, In_1Q]

model1 = np.empty([100, 72, 0])

for file in file_list:
    file = np.array(file)
    model1=np.dstack((model1, file))

file_list2 = [Rn_1A, Rn_1B, Rn_1C, Rn_1DA,
             Rn_1DB, Rn_1DC, Rn_1E, Rn_1F,
             Rn_1G, Rn_1H, Rn_1I, Rn_1J,
             Rn_1K, Rn_1L, Rn_1M, Rn_1N, 
             Rn_1O, Rn_1P, Rn_1Q]

model2 = np.empty([100, 72, 0])

for file in file_list2:
    file = np.array(file)
    model2=np.dstack((model2, file))

file_list3 = [En_1A, En_1B, En_1C, En_1DA,
             En_1DB, En_1DC, En_1E, En_1F,
             En_1G, En_1H, En_1I, En_1J,
             En_1K, En_1L, En_1M, En_1N, 
             En_1O, En_1P, En_1Q]

model3 = np.empty([100, 72, 0])

for file in file_list3:
    file = np.array(file)
    model3=np.dstack((model3, file))

file_list4 = [Sn_1A, Sn_1B, Sn_1C, Sn_1DA,
             Sn_1DB, Sn_1DC, Sn_1E, Sn_1F,
             Sn_1G, Sn_1H, Sn_1I, Sn_1J,
             Sn_1K, Sn_1L, Sn_1M, Sn_1N, 
             Sn_1O, Sn_1P, Sn_1Q]

model4 = np.empty([100, 72, 0])

for file in file_list4:
    file = np.array(file)
    model4=np.dstack((model4, file))

file_list5 = [Incid_1A, Incid_1B, Incid_1C, Incid_1DA,
             Incid_1DB, Incid_1DC, Incid_1E, Incid_1F,
             Incid_1G, Incid_1H, Incid_1I, Incid_1J,
             Incid_1K, Incid_1L, Incid_1M, Incid_1N, 
             Incid_1O, Incid_1P, Incid_1Q]

model5 = np.empty([100, 72, 0])

for file in file_list5:
    file = np.array(file)
    model5=np.dstack((model5, file))

# Summary: 
# model 1 = In 
# model 2 = Rn
# model 3 = En
# model 4 = Sn
# model 5 = Incidences

## Seroprevalence (global) = (Rn)/(In + Rn + En + Sn)
numerator = model2
denominator = model1 + model2 + model3 + model4

## Make sure there are no zeroes in denominator by making 0 a really small number
denominator = np.where(denominator == 0, 1e-10, denominator)

global_seroprevalence = numerator/denominator

# We only care about week 52 (1 year)
week_52_data = global_seroprevalence[:, 51, :] # yields 100 reps x 14400 systems matrix.

# final shape of week_52_data would be 14400 x 100
# flatten this and transpose
# final shape (1440000,)
flat_global_sero_prev = week_52_data.T.flatten()
flat_global_sero_prev = pd.DataFrame(flat_global_sero_prev)

## Calculations for Peak Cases to keep
## # of sims consistent
num_reps = model5.shape[0]
num_weeks = model5.shape[1]
num_systems = model5.shape[2]
peak_cases = np.zeros((num_reps, num_systems))

# Loop through each system
for system_idx in range(num_systems):  
    # Loop through each rep
    for rep_idx in range(num_reps):
        # Sum values across all weeks for the current system and rep
        weekly_sum = np.sum(model5[rep_idx, :, system_idx])
        
        # If the sum is greater than 10, there is an outbreak (<= 10 means no outbreak)
        if weekly_sum > 10:
            # Find the week with the highest value for this rep and system
            week_max = np.argmax(model5[rep_idx, :, system_idx])
            peak_value = model5[rep_idx, week_max, system_idx]
            
            # Only update peak_cases if the peak value is greater than 1
            # This is checking for long term stuttering
            if peak_value > 1: 
                peak_cases[rep_idx, system_idx] = peak_value

# final shape of peak_cases would be 14400 x 100
# flatten this and transpose
# final shape (1440000,)
flat_peak_cases = peak_cases.T.flatten()
flat_peak_cases = pd.DataFrame(flat_peak_cases)

# Correctly repeats the parameters to match up with the flattened output data
parameters_df = pd.read_csv('../FullSetOfParameters.csv')  # Shape: 14400 x 7
parameters_repeated = pd.DataFrame(np.repeat(parameters_df.values, 100, axis=0))

# Groups the output data (given value of 1-14400 to place it with its system)
system_id = np.repeat(np.arange(1, parameters_repeated.shape[0] // 100 + 1), 100)
parameters_repeated['System_ID'] = system_id

data_df = pd.concat([parameters_repeated, flat_peak_cases, flat_global_sero_prev], axis=1, ignore_index=True)
print(data_df.shape)

new_columns = ['Movement', 'HostDensity', 'InfectPeriodLive', 'InfectPeriodCarcass', 
               'IncubationPeriod','ProportionRecover', 'DaysToRecovery', 'System_ID', 'PeakCases', 'GlobalSeroPrev']
    
data_df.columns = new_columns # had to re-add column names

# Filter out rows where 'PeakCases' is 0
df_filtered = data_df[data_df.iloc[:, 8] !=0]

# Create the 6 dataframes based on combinations of Movement and HostDensity
# Save Output data to file
df_lowdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 1.5)]
df_lowdens_movement1.iloc[:, 9].to_csv('LowDens-LowMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_lowdens_movement1.shape)

df_meddens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 3)]
df_meddens_movement1.iloc[:, 9].to_csv('MedDens-LowMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_meddens_movement1.shape)

df_highdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 5)]
df_highdens_movement1.iloc[:, 9].to_csv('HighDens-LowMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_highdens_movement1.shape)

df_lowdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 1.5)]
df_lowdens_movement2.iloc[:, 9].to_csv('LowDens-HighMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_lowdens_movement2.shape)

df_meddens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 3)]
df_meddens_movement2.iloc[:, 9].to_csv('MedDens-HighMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_meddens_movement2.shape)

df_highdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 5)]
df_highdens_movement2.iloc[:, 9].to_csv('HighDens-HighMove-GlobalSeroPrev-at52.csv', index=False, header=False)
print(df_highdens_movement2.shape)

# Define combinations of Movement and HostDensity
movement_options = [1, 2]
host_density_options = [1.5, 3, 5]

# Loop through each combination and save to separate files
for movement in movement_options:
    for density in host_density_options:
        # Filter the dataframe for the current combination of Movement and HostDensity
        filtered_df = df_filtered[(df_filtered['Movement'] == movement) & (df_filtered['HostDensity'] == density)]
        
        # Create a descriptive filename based on the parameters
        filename = f"Movement_{movement}_HostDensity_{density}.csv"
        
        # Save the filtered dataframe to a CSV file
        filtered_df.to_csv(filename, index=False)


# Final shape - same order from above
# (183645, 9)
# (189540, 9)
# (184858, 9)
# (67289, 9)
# (85811, 9)
# (90589, 9)