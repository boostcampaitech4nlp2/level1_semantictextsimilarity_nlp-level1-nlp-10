from sentence_transformers import SentenceTransformer, models
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import batch_to_device
from sklearn.metrics.pairwise import paired_cosine_distances
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from train_SBERT import convert2data, calculate_cosine_scores
import numpy as np
import os
import sys
from tqdm import tqdm


def create_submision(model_path, save_path=None):
    model = SentenceTransformer.load(model_path)
    test_path = '../data/dev.csv'
    test_df = pd.read_csv(test_path)
    test_data = convert2data(test_df, is_label=False)
    cosine_scores = calculate_cosine_scores(model, *test_data)
    cosine_scores = np.array(cosine_scores)
    ids = test_df['id'].values
    if save_path is None:
        save_path = model_path
    save_result(ids, cosine_scores, save_path)
    
def save_result(ids, result, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'submission.csv'), 'w') as f:
        for idx, data in zip(ids, result):
            text = f'{idx},{data:.1f}\n'
            f.write(text)
            

'''
실행 : python3 inference_SBERT.py model_path save_path=None
ex) python3 inference_SBERT.py ../data/sbert/klue-roberta-large/2022-11-02/ 
ex) python3 inference_SBERT.py ../data/sbert/klue-roberta-large/2022-11-02/ ./inferences/

'''
if __name__ == '__main__':
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    else:
        save_path = None
    create_submision(sys.argv[1], save_path)