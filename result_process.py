import pickle
import os
import pandas as pd
import re
import shutil

dataset = 'physionet'

# Path to the root directory where the search should start
directory_path = f'./log/{dataset}/'

file_dir = os.path.join('results')
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
    
task_result = os.path.join(file_dir, dataset)
if not os.path.exists(task_result):
    os.makedirs(task_result)
    os.makedirs(task_result + '/valid/')

exps = os.listdir(directory_path)

for model_name in exps:
    try:
        path = f'{directory_path}/{model_name}/'
        results = pd.DataFrame()
        for i in os.listdir(path):
            try:
                result = pickle.load(open(path + i + '/kfold_best.pkl', 'rb'))
                subresults = pd.DataFrame()
                for key, value in result.items():
                    if 'bets_valid' in key:
                        value['model']=i
                        value['fold'] = key
                        subresults = pd.concat([subresults, pd.DataFrame([value])])
                overall = subresults[subresults.columns.drop('fold').drop('model')].mean().to_frame().T
                overall['model'] = i + '_overall_valid'
                subresults = pd.concat([subresults, overall])
                results = pd.concat([results, subresults])
            except:
                print('Not finished: ', path + i)
                continue
        results.sort_values(['model'],ascending=[False])
        results.to_csv('{}/valid/{}.csv'.format(task_result, model_name), index= False)
    except:
        continue