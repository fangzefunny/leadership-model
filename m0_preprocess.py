import os 
import pickle
import numpy as np 
import pandas as pd 

# the path to the current file 
pth = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f'{pth}/data') is False: os.makedirs(f'{pth}/data')

def preprocess_data():
    # load data 
    data_pth = f'{pth}/raw_data/leadership_data/'
    sub_folders = [f for f in os.listdir(data_pth) if os.path.isdir(os.path.join(data_pth, f))]
    data, data_type1, data_type2 = [], [], []
    data_for_fit, data_type1_for_fit, data_type2_for_fit = {}, {}, {}
    for sub_folder in sub_folders:
        block_data, block_data_type1, block_data_type2 = {}, {}, {}
        # Check if the number of files in the sub_folder is less than 5
        files_in_folder = [f for f in os.listdir(f'{data_pth}/{sub_folder}') if f.startswith('run') and f.endswith('.txt')]
        # Sort the files to ensure consistent order
        files_in_folder.sort(key=lambda x: int(x.replace('run', '').replace('.txt', '')))
        # Check if the folder contains exactly the required run files
        required_files = ['run1.txt', 'run2.txt', 'run3.txt', 'run4.txt']
        if sorted(files_in_folder) != sorted(required_files):
            continue
        for i in range(1, 5):
            fname = f'{data_pth}/{sub_folder}/run{i}.txt'
            datum = pd.read_csv(fname, sep='\t')
            datum.rename(columns={'SubjectiveValue': 'ObjectiveValue'}, inplace=True)
            datum.drop(columns=['Unnamed: 9'], inplace=True)
            # Skip empty dataframes
            if datum.shape[0] == 0: continue
            # reverse engineering the z_gain, z_loss, z_uncertain
            def reverse_engineering(row):
                uncertain = row['Uncertainty']
                gap = row['ObjectiveValue']
                remain = 10-uncertain
                z_gain = (gap+remain)/2
                z_loss = remain - z_gain
                return z_gain, z_loss
            datum[['Zgain', 'Zloss']] = datum.apply(reverse_engineering, axis=1, result_type='expand')
            # turn the Fishing action to index
            datum['Fishing'] = datum['Fishing'].apply(lambda x: x-1)
            # add index 
            datum['run_id'] = i
            datum['sub_id'] = sub_folder.split('_')[0]  
            data.append(datum)
            data_type1.append(datum.query('Type==1').copy())
            data_type2.append(datum.query('Type==2').copy())
            block_data[i] = datum
            block_data_type1[i] = datum.query('Type==1').copy()
            block_data_type2[i] = datum.query('Type==2').copy()
        data_for_fit[sub_folder] = block_data
        data_type1_for_fit[sub_folder] = block_data_type1
        data_type2_for_fit[sub_folder] = block_data_type2
        print(f'{sub_folder} has {len(files_in_folder)} runs')

    # data for analysis 
    data = pd.concat(data)
    data.to_csv(f'{pth}/data/leadership_data.csv', index=False)
    data_type1 = pd.concat(data_type1)
    data_type1.to_csv(f'{pth}/data/leadership_data_type1.csv', index=False)
    data_type2 = pd.concat(data_type2)
    data_type2.to_csv(f'{pth}/data/leadership_data_type2.csv', index=False)
    # data for fit
    fname = f'{pth}/data/leadership_data.pkl'
    with open(fname, 'wb') as handle: pickle.dump(data_for_fit, handle)
    fname = f'{pth}/data/leadership_data_type1.pkl'
    with open(fname, 'wb') as handle: pickle.dump(data_type1_for_fit, handle)
    fname = f'{pth}/data/leadership_data_type2.pkl'
    with open(fname, 'wb') as handle: pickle.dump(data_type2_for_fit, handle)

if __name__ == '__main__':

    preprocess_data()