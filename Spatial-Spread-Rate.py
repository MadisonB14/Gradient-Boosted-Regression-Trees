import numpy as np # type: ignore
import h5py # type: ignore
import warnings
import pandas as pd # type: ignore
import csv
import statistics

warnings.filterwarnings("ignore", category=DeprecationWarning)

#path where script and data should be run from
#os.chdir('OneDrive - USDA/Documents/Research/ASF_Project/AllSimulationData')

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

#####################################################################
#### Structure of 2nd Set of Mat files by column with definition ####
################### Sensitivity_2 ###################################
# linsprIEC - # of alive individuals within I, E & C polygon
# AliveOutIEC - # of alive individuals outside I, E & C polygon
# AreaIEC - area of current polygon around all I, E & C
# CellsIEC - # of grid cells with I, E or C
# DeadInIEC - # of dead individuals within I, E & C polygon
# DeadOutIEC - # of dead individuals outside I, E & C polygon
# LinearSpreadIEC - furthest I, E or C from index case
#####################################################################

rep2A = h5py.File('../Sensitivity2_1to1154.mat','r')     
linspr_1A = rep2A.get('AreaIEC')
linspr_1A = np.delete(linspr_1A, range(1154,14400), axis=2)
# print("Shape of rep2A:", linspr_1A.shape)

rep2B = h5py.File('../Sensitivity2_1155to1355.mat','r')     
linspr_1B = rep2B.get('AreaIEC')
linspr_1B = np.delete(linspr_1B, range(201,7845), axis=2)
# print("Shape of rep2B:", linspr_1B.shape)
# print(linspr_1B[:, :, 7000])

rep2C = h5py.File('../Sensitivity2_1356to2280.mat','r')     
linspr_1C = rep2C.get('AreaIEC')
linspr_1C = np.delete(linspr_1C, range(925,7644), axis=2)
# print("Shape of rep2C:", linspr_1C.shape)

rep2DA = h5py.File('../Sensitivity2_2281and4081.mat','r')     
linspr_1DA = rep2DA.get('AreaIEC')
linspr_1DA = np.delete(linspr_1DA, 1, axis=2)
# print("Shape of rep2DA:", linspr_1DA.shape)

rep2DB = h5py.File('../Sensitivity2_2282to4080.mat','r')     
linspr_1DB = rep2DB.get('AreaIEC')
linspr_1DB = np.delete(linspr_1DB, range(1799,6718), axis=2)
# print("Shape of rep2DB:", linspr_1DB.shape)

rep2DC = h5py.File('../Sensitivity2_2281and4081.mat','r')     
linspr_1DC = rep2DC.get('AreaIEC')
linspr_1DC = np.delete(linspr_1DC, 0, axis=2)
# print("Shape of rep2DC:", linspr_1DC.shape)

rep2E = h5py.File('../Sensitivity2_4082to5293.mat','r')     
linspr_1E = rep2E.get('AreaIEC')
linspr_1E= np.delete(linspr_1E, range(1212,4918), axis=2)
# print("Shape of rep2E:", linspr_1E.shape)

rep2F = h5py.File('../Sensitivity2_5294to6241.mat','r')     
linspr_1F = rep2F.get('AreaIEC')
linspr_1F = np.delete(linspr_1F, range(948,3706), axis=2)
# print("Shape of rep2F:", linspr_1F.shape)

rep2G = h5py.File('../Sensitivity2_6242to6961.mat','r')     
linspr_1G = rep2G.get('AreaIEC')
linspr_1G = np.delete(linspr_1G, range(720,2758), axis=2)
# print("Shape of rep2G:", linspr_1G.shape)

rep2H = h5py.File('../Sensitivity2_6962to8281.mat','r')     
linspr_1H = rep2H.get('AreaIEC')
linspr_1H = np.delete(linspr_1H, range(1320,2038), axis=2)
# print("Shape of rep2H:", linspr_1H.shape)

rep2I = h5py.File('../Sensitivity2_8282to8999.mat','r')     
linspr_1I = rep2I.get('AreaIEC')
# print("Shape of rep2I:", linspr_1I.shape)

rep2J = h5py.File('../Sensitivity2_9000to9889.mat','r')     
linspr_1J = rep2J.get('AreaIEC')
linspr_1J = np.delete(linspr_1J, range(890,3000), axis=2)
# print("Shape of rep2J:", linspr_1J.shape)

rep2K = h5py.File('../Sensitivity2_9890to10561.mat','r')     
linspr_1K = rep2K.get('AreaIEC')
linspr_1K = np.delete(linspr_1K, range(672,2110), axis=2)
# print("Shape of rep2K:", linspr_1K.shape)

rep2L = h5py.File('../Sensitivity2_10562to11065.mat','r')     
linspr_1L = rep2L.get('AreaIEC')
linspr_1L = np.delete(linspr_1L, range(504,1438), axis=2)
# print("Shape of rep2L:", linspr_1L.shape)

rep2M = h5py.File('../Sensitivity2_11066to11380.mat','r')     
linspr_1M = rep2M.get('AreaIEC')
linspr_1M = np.delete(linspr_1M, range(315,934), axis=2)
# print("Shape of rep2M:", linspr_1M.shape)

rep2N = h5py.File('../Sensitivity2_11381to11499.mat','r')     
linspr_1N = rep2N.get('AreaIEC')
# print("Shape of rep2N:", linspr_1N.shape)

rep2O = h5py.File('../Sensitivity2_11500to11999.mat','r')     
linspr_1O = rep2O.get('AreaIEC')
# print("Shape of rep2O:", linspr_1O.shape)

rep2P = h5py.File('../Sensitivity2_12000to13681.mat','r')     
linspr_1P = rep2P.get('AreaIEC')
linspr_1P = np.delete(linspr_1P, range(1682,2401), axis=2)
# print("Shape of rep2P:", linspr_1P.shape)

rep2Q = h5py.File('../Sensitivity2_13682to14400.mat','r')     
linspr_1Q = rep2Q.get('AreaIEC')
# print("Final z dimension:", linspr_1Q[0, 1, :])
# print("Shape of rep2Q:", linspr_1Q.shape)

file_list = [linspr_1A, linspr_1B, linspr_1C, linspr_1DA,
             linspr_1DB, linspr_1DC, linspr_1E, linspr_1F,
             linspr_1G, linspr_1H, linspr_1I, linspr_1J,
             linspr_1K, linspr_1L, linspr_1M, linspr_1N, 
             linspr_1O, linspr_1P, linspr_1Q]

model2 = np.empty([100, 72, 0])

for file in file_list:
    file = np.array(file)
    model2=np.dstack((model2, file))
    # print("New shape of model:", model2.shape)

## Calculate rate of linear spread
weekly_change = np.diff(model2, axis=1)
median_weekly_change = np.median(weekly_change, axis=1)

# final shape of week_52_data would be 14400 x 100
# flatten this and transpose
# final shape (1440000,)
flat_spatial_spread = median_weekly_change.T.flatten()
flat_spatial_spread = pd.DataFrame(flat_spatial_spread)

#### Process Peak Cases Data so we stay consistent
#### on systems that have outbreaks
num_reps = model1.shape[0]
num_weeks = model1.shape[1]
num_systems = model1.shape[2]
peak_cases = np.zeros((num_reps, num_systems))

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

data_df = pd.concat([parameters_repeated, flat_peak_cases, flat_spatial_spread], axis=1, ignore_index=True)
print(data_df.shape)

new_columns = ['Movement', 'HostDensity', 'InfectPeriodLive', 'InfectPeriodCarcass', 
               'IncubationPeriod','ProportionRecover', 'DaysToRecovery', 'PeakCases', 'SpatialSpreadRate']
    
data_df.columns = new_columns # had to re-add column names

# Filter out rows where 'PeakCases' is 0
df_filtered = data_df[data_df.iloc[:, 7] !=0]

# Create the 6 dataframes based on combinations of Movement and HostDensity
# Save Output data to file
df_lowdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 1.5)]
df_lowdens_movement1.iloc[:, 8].to_csv('LowDens-LowMove-RateofSpread-Median.csv', index=False, header=False)
print(df_lowdens_movement1.shape)

df_meddens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 3)]
df_meddens_movement1.iloc[:, 8].to_csv('MedDens-LowMove-RateofSpread-Median.csv', index=False, header=False)
print(df_meddens_movement1.shape)

df_highdens_movement1 = df_filtered[(df_filtered['Movement'] == 1) & (df_filtered['HostDensity'] == 5)]
df_highdens_movement1.iloc[:, 8].to_csv('HighDens-LowMove-RateofSpread-Median.csv', index=False, header=False)
print(df_highdens_movement1.shape)

df_lowdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 1.5)]
df_lowdens_movement2.iloc[:, 8].to_csv('LowDens-HighMove-RateofSpread-Median.csv', index=False, header=False)
print(df_lowdens_movement2.shape)

df_meddens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 3)]
df_meddens_movement2.iloc[:, 8].to_csv('MedDens-HighMove-RateofSpread-Median.csv', index=False, header=False)
print(df_meddens_movement2.shape)

df_highdens_movement2 = df_filtered[(df_filtered['Movement'] == 2) & (df_filtered['HostDensity'] == 5)]
df_highdens_movement2.iloc[:, 8].to_csv('HighDens-HighMove-RateofSpread-Median.csv', index=False, header=False)
print(df_highdens_movement2.shape)


