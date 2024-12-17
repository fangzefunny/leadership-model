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
    data = []
    data_for_fit = {}
    for sub_folder in sub_folders:
        block_data = {}
        for i in range(1, 5):
            fname = f'{data_pth}/{sub_folder}/run{i}.txt'
            datum = pd.read_csv(fname, sep='\t')
            datum.drop(columns=['Unnamed: 9'], inplace=True)
            # reverse engineering the z_gain, z_loss, z_uncertain
            def reverse_engineering(row):
                uncertain = row['Uncertainty']
                gap = row['SubjectiveValue']
                remain = 10-uncertain
                z_gain = (gap+remain)/2
                z_loss = remain - z_gain
                return z_gain, z_loss
            datum[['Zgain', 'Zloss']] = datum.apply(reverse_engineering, axis=1, result_type='expand')
            # turn the Fishing action to index
            datum['Fishing'] = datum['Fishing'].apply(lambda x: x-1)
            # remove rows with Fishing > 5
            datum = datum.query('Fishing < 5').copy()
            # add index 
            datum['run_id'] = i
            datum['sub_id'] = sub_folder.split('_')[0]  
            data.append(datum)
            block_data[i] = datum
        data_for_fit[sub_folder] = block_data

    # data for analysis 
    data = pd.concat(data)
    data.to_csv(f'{pth}/data/leadership_data.csv', index=False)
    # data for fit
    fname = f'{pth}/data/leadership_data.pkl'
    with open(fname, 'wb') as handle: pickle.dump(data_for_fit, handle)

if __name__ == '__main__':

    preprocess_data()