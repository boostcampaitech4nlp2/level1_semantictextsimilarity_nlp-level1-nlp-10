import os, random, torch
import numpy as np

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
     
def make_file_name(model_name, format,  version='v0'): 
    file_name = f'{model_name}_{version}.{format}'
    return file_name