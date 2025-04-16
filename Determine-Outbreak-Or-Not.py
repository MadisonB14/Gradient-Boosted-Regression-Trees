import numpy as np # type: ignore
import h5py # type: ignore
import warnings
import pandas as pd # type: ignore
import csv
import statistics

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
incid_1A = rep1A.get('Incidence')
incid_1A = np.delete(incid_1A, range(1154,14400), axis=2)
# print("Shape of rep1A:", incid_1A.shape)

rep1B = h5py.File('../Sensitivity1_1155to1355.mat','r')     
incid_1B = rep1B.get('Incidence')
incid_1B = np.delete(incid_1B, range(201,7845), axis=2)
# print("Shape of rep1B:", incid_1B.shape)
# print(incid_1B[:, :, 7000])

rep1C = h5py.File('../Sensitivity1_1356to2280.mat','r')     
incid_1C = rep1C.get('Incidence')
incid_1C = np.delete(incid_1C, range(925,7644), axis=2)
# print("Shape of rep1C:", incid_1C.shape)

rep1DA = h5py.File('../Sensitivity1_2281and4081.mat','r')     
incid_1DA = rep1DA.get('Incidence')
incid_1DA = np.delete(incid_1DA, 1, axis=2)
# print("Shape of rep1DA:", incid_1DA.shape)

rep1DB = h5py.File('../Sensitivity1_2282to4080.mat','r')     
incid_1DB = rep1DB.get('Incidence')
incid_1DB = np.delete(incid_1DB, range(1799,6718), axis=2)
# print("Shape of rep1DB:", incid_1DB.shape)

rep1DC = h5py.File('../Sensitivity1_2281and4081.mat','r')     
incid_1DC = rep1DC.get('Incidence')
incid_1DC = np.delete(incid_1DC, 0, axis=2)
# print("Shape of rep1DC:", incid_1DC.shape)

rep1E = h5py.File('../Sensitivity1_4082to5293.mat','r')     
incid_1E = rep1E.get('Incidence')
incid_1E= np.delete(incid_1E, range(1212,4918), axis=2)
# print("Shape of rep1E:", incid_1E.shape)

rep1F = h5py.File('../Sensitivity1_5294to6241.mat','r')     
incid_1F = rep1F.get('Incidence')
incid_1F = np.delete(incid_1F, range(948,3706), axis=2)
# print("Shape of rep1F:", incid_1F.shape)

rep1G = h5py.File('../Sensitivity1_6242to6961.mat','r')     
incid_1G = rep1G.get('Incidence')
incid_1G = np.delete(incid_1G, range(720,2758), axis=2)
# print("Shape of rep1G:", incid_1G.shape)

rep1H = h5py.File('../Sensitivity1_6962to8281.mat','r')     
incid_1H = rep1H.get('Incidence')
incid_1H = np.delete(incid_1H, range(1320,2038), axis=2)
# print("Shape of rep1H:", incid_1H.shape)

rep1I = h5py.File('../Sensitivity1_8282to8999.mat','r')     
incid_1I = rep1I.get('Incidence')
# print("Shape of rep1I:", incid_1I.shape)

rep1J = h5py.File('../Sensitivity1_9000to9889.mat','r')     
incid_1J = rep1J.get('Incidence')
incid_1J = np.delete(incid_1J, range(890,3000), axis=2)
# print("Shape of rep1J:", incid_1J.shape)

rep1K = h5py.File('../Sensitivity1_9890to10561.mat','r')     
incid_1K = rep1K.get('Incidence')
incid_1K = np.delete(incid_1K, range(672,2110), axis=2)
# print("Shape of rep1K:", incid_1K.shape)

rep1L = h5py.File('../Sensitivity1_10562to11065.mat','r')     
incid_1L = rep1L.get('Incidence')
incid_1L = np.delete(incid_1L, range(504,1438), axis=2)
# print("Shape of rep1L:", incid_1L.shape)

rep1M = h5py.File('../Sensitivity1_11066to11380.mat','r')     
incid_1M = rep1M.get('Incidence')
incid_1M = np.delete(incid_1M, range(315,934), axis=2)
# print("Shape of rep1M:", incid_1M.shape)

rep1N = h5py.File('../Sensitivity1_11381to11499.mat','r')     
incid_1N = rep1N.get('Incidence')
# print("Shape of rep1N:", incid_1N.shape)

rep1O = h5py.File('../Sensitivity1_11500to11999.mat','r')     
incid_1O = rep1O.get('Incidence')
# print("Shape of rep1O:", incid_1O.shape)

rep1P = h5py.File('../Sensitivity1_12000to13681.mat','r')     
incid_1P = rep1P.get('Incidence')
incid_1P = np.delete(incid_1P, range(1682,2401), axis=2)
# print("Shape of rep1P:", incid_1P.shape)

rep1Q = h5py.File('../Sensitivity1_13682to14400.mat','r')     
incid_1Q = rep1Q.get('Incidence')
# print("Final z dimension:", incid_1Q[0, 1, :])
# print("Shape of rep1Q:", incid_1Q.shape)

file_list = [incid_1A, incid_1B, incid_1C, incid_1DA,
             incid_1DB, incid_1DC, incid_1E, incid_1F,
             incid_1G, incid_1H, incid_1I, incid_1J,
             incid_1K, incid_1L, incid_1M, incid_1N, 
             incid_1O, incid_1P, incid_1Q]

model1 = np.empty([100, 72, 0])

for file in file_list:
    file = np.array(file)
    model1=np.dstack((model1, file))
    # print("New shape of model:", model1.shape)

num_reps = model1.shape[0]
num_weeks = model1.shape[1]
num_systems = model1.shape[2]

# Initialize a 2D array to store the outbreak results
# Shape: (num_reps, num_systems)
classifications = np.zeros((num_reps, num_systems))

# Loop through each system
for system_idx in range(num_systems):
    # Loop through each rep
    for rep_idx in range(num_reps):
        # Sum values across all weeks for the current system and rep
        weekly_sum = np.sum(model1[rep_idx, :, system_idx])
        # print("Weekly sum:", weekly_sum)     
        
        # Check for "no outbreak" conditions:
        # 1. Weekly sum <= 10
        # 2. All values for the current system and rep are <= 1 (no significant incidence)
        if weekly_sum <= 10 or np.all(model1[rep_idx, :, system_idx] <= 1):
            classifications[rep_idx, system_idx] = 0  # No outbreak
        else:
            classifications[rep_idx, system_idx] = 1  # Outbreak

## Double check the number of no outbreak
num_zeros = np.size(classifications) - np.count_nonzero(classifications)

flat_classes = classifications.T.flatten()

df = pd.DataFrame(flat_classes)
df.to_csv('Outbreaks_Classified.csv', index=False, header=False)

# Correctly repeats the parameters to match up with the flattened output data
parameters_df = pd.read_csv('../FullSetOfParameters.csv')  # Shape: 14400 x 7
parameters_repeated = pd.DataFrame(np.repeat(parameters_df.values, 100, axis=0))

# Groups the output data (given value of 1-14400 to place it with its system)
system_id = np.repeat(np.arange(1, parameters_repeated.shape[0] // 100 + 1), 100)
parameters_repeated['System_ID'] = system_id

data_df = pd.concat([parameters_repeated, df], axis=1, ignore_index=True)
print(data_df.shape)

new_columns = ['Movement', 'HostDensity', 'InfectPeriodLive', 'InfectPeriodCarcass', 
               'IncubationPeriod','ProportionRecover', 'DaysToRecovery', 'System_ID', 'Outbreak']
    
data_df.columns = new_columns # had to re-add column names

# Create the 6 dataframes based on combinations of Movement and HostDensity
# Save Output data to file
df_lowdens_movement1 = data_df[(data_df['Movement'] == 1) & (data_df['HostDensity'] == 1.5)]
df_lowdens_movement1.iloc[:, 8].to_csv('LowDens-LowMove-outbreakprob.csv', index=False, header=False)
print(df_lowdens_movement1.shape)

df_meddens_movement1 = data_df[(data_df['Movement'] == 1) & (data_df['HostDensity'] == 3)]
df_meddens_movement1.iloc[:, 8].to_csv('MedDens-LowMove-outbreakprob.csv', index=False, header=False)
print(df_meddens_movement1.shape)

df_highdens_movement1 = data_df[(data_df['Movement'] == 1) & (data_df['HostDensity'] == 5)]
df_highdens_movement1.iloc[:, 8].to_csv('HighDens-LowMove-outbreakprob.csv', index=False, header=False)
print(df_highdens_movement1.shape)

df_lowdens_movement2 = data_df[(data_df['Movement'] == 2) & (data_df['HostDensity'] == 1.5)]
df_lowdens_movement2.iloc[:, 8].to_csv('LowDens-HighMove-outbreakprob.csv', index=False, header=False)
print(df_lowdens_movement2.shape)

df_meddens_movement2 = data_df[(data_df['Movement'] == 2) & (data_df['HostDensity'] == 3)]
df_meddens_movement2.iloc[:, 8].to_csv('MedDens-HighMove-outbreakprob.csv', index=False, header=False)
print(df_meddens_movement2.shape)

df_highdens_movement2 = data_df[(data_df['Movement'] == 2) & (data_df['HostDensity'] == 5)]
df_highdens_movement2.iloc[:, 8].to_csv('HighDens-HighMove-outbreakprob.csv', index=False, header=False)
print(df_highdens_movement2.shape)

# Define combinations of Movement and HostDensity
movement_options = [1, 2]
host_density_options = [1.5, 3, 5]

# Loop through each combination and save to separate files
for movement in movement_options:
    for density in host_density_options:
        # Filter the dataframe for the current combination of Movement and HostDensity
        filtered_df = data_df[(data_df['Movement'] == movement) & (data_df['HostDensity'] == density)]
        
        # Create a descriptive filename based on the parameters
        filename = f"OutbreakProb_Params_Movement_{movement}_HostDensity_{density}.csv"
        
        # Save the filtered dataframe to a CSV file
        filtered_df.to_csv(filename, index=False)

