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

def check_param(modelname, batchsize, maxepoch, shuffle, lr, seed):
    print('-'*40)
    print('********CHECK PARAMETERS********')
    print('MODEL NAME', modelname)
    print('BATCH SIZE', batchsize)
    print('MAX EPOCH', maxepoch)
    print('SHUFFLE', shuffle)
    print('LEARNING RATE', lr)
    print('SEED', seed)
    print('-'*40)