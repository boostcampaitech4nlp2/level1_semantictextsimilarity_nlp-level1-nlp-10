import os, random, torch, math
import numpy as np
import pandas as pd

def set_seed(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def setdir(dirpath, dirname=None, reset=True):
    from shutil import rmtree
    filepath = os.path.join(dirpath, dirname) if dirname else dirpath      
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    elif reset:
        print(f"reset directory : {dirname}")
        rmtree(filepath)
        os.mkdir(filepath)
    return filepath 

def check_params(model_name, batch_size, max_epoch, shuffle, lr, seed):
    print('-'*40)
    print('********CHECK PARAMETERS********')
    print('MODEL NAME    |', model_name)
    print('BATCH SIZE    |', batch_size)
    print('MAX EPOCH     |', max_epoch)
    print('SHUFFLE       |', shuffle)
    print('LEARNING RATE |', lr)
    print('SEED          |', seed)
    print('-'*40)

def convert_boolean_args(arg):
    if arg in ['true', 'True', 't', 'T']:
        return True
    return False

def make_file_name(model_name, format,  version='v0'): 
    file_name = f'{model_name}_{version}.{format}'
    return file_name

def hard_voting(load_dir, submission_files, version='v0'):
    submission = pd.read_csv('../data/sample_submission.csv')
    for i, submission_file in enumerate(submission_files):
        submission_path = os.path.join(load_dir, submission_file)
        df = pd.read_csv(submission_path)
        if i == 0:
            submission['target'] = df['target']
        else:
            submission['target'] += df['target']
    submission['target'] = round(submission['target']/len(submission_files), 1)
    save_name = make_file_name('hard_voting', format='csv', version=version)
    save_path = os.path.join(load_dir, save_name)
    submission.to_csv(save_path, index=False)
        