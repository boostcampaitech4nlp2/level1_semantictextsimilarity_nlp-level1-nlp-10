import sys
import os
import math
from datetime import datetime
from subprocess import check_output
from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model
from sts.utils import set_seed, setdir, check_params, make_file_name
import torch
import torchmetrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import batch_to_device
from sklearn.metrics.pairwise import paired_cosine_distances


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(args.data_dir, args.model_dir, reset=False)
    batch_size = int(args.batch_size)
    epochs = int(args.max_epoch)
    
    #argparser로 받아온 parameter을 terminal에 출력하여 확인할 수 있게 합니다
    check_params(args.model_name, args.batch_size, args.max_epoch, args.shuffle, args.learning_rate, args.seed)
    
    # 1. 기존 dataloader를 사용하지 않고 dataloder를 만듭니다.
    #    기존 dataloader가 [SEP]토큰을 사이에 두고 두 문장을 연결해서 하나의 문장을 만들어주는 건데,
    #    SBERT에서는 두 문장을 각각 따로 입력해야 하기 때문에 기존 dataloader를 사용하지 않았습니다.
    train_df = pd.read_csv(args.train_path)
    dev_df = pd.read_csv(args.dev_path)
    train_examples = make_sts_input_example(train_df)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    dev_data = convert2data(dev_df)
    
    
    # 2. embedding 모델과 pooling 모델을 생성하고 SentenceTransformer로 연결합니다.
    model_name = args.model_name
    #model_name = 'klue/roberta-large'
    embedding_model = models.Transformer(
        model_name_or_path=model_name,
        max_seq_length=256,
        do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[embedding_model, pooling_model])
    
    if args.wandb:
        wandb.login()
    
    # 3. 모델 저장 경로를 지정하고 model.fit()으로 학습합니다.
    #    losses.CosineSimilarityLoss() 함수를 사용하면 CosineSimilarityLoss을 사용하는 모델을 리턴합니다.
    #    그러나 train시에만 CosineSimilairtyLoss로 감싸서 사용하고,
    #    이후 inference 시에는 감싼 모델이 아니라 감싸지지 않은 모델을 사용하는 것 같습니다.
    root_path = '../data/sbert/'
    save_path = root_path + model_name.replace('/', '-') + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_epochs' + str(epochs)

    train_loss = losses.CosineSimilarityLoss(model=model)
    warmup_steps = math.ceil(len(train_examples) * epochs / batch_size * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        #evaluator=dev_evaluator,
        epochs=epochs,
        #evaluation_steps=int(len(train_dataloader)*0.1),
        warmup_steps=warmup_steps,
        output_path=save_path
    )
    
    # 4. 여기에 dev.csv로 pearson 점수를 계산하는 부분이 필요합니다!
    score = evaluate(model, *dev_data, batch_size=batch_size)
    print(f'test_pearson : {score}')
    
    
# input data를 InputExample 데이터들의 리스트로 만들어서 리턴합니다.
# InputExample은 StentenceTrnasformer로 입력되는 데이터 형식입니다.
def make_sts_input_example(dataset):
    input_examples = []
    for i, data in tqdm(dataset.iterrows(), desc='SBERT input_examples', total=len(dataset)):
        st1 = data['sentence_1']
        st2 = data['sentence_2']
        score = (data['label']) / 5.0
        input_examples.append(InputExample(texts=[st1, st2], label=score))
    return input_examples

# dev.csv로 pearson 점수를 계산하는 함수를 만들어보는 중입니다.
# 현재 밑에 load_test() 함수 안에서 사용되고 있습니다.
def evaluate(model, st1, st2, labels, batch_size, save_result=False):
    #dataloader.collate_fn = model.smart_batching_collate
    emb1 = model.encode(st1, batch_size=batch_size, convert_to_numpy=True)
    emb2 = model.encode(st2, batch_size=batch_size, convert_to_numpy=True)
    
    # 1에서 cosine_distance를 빼는 것으로 cosine_score를 구합니다.
    # 그리고 데이터의 label 부분과 수치를 맞춰주기 위해 5를 곱했고, 0보다 작은 데이터는 0으로 설정했습니다.
    cosine_scores = 1 - (paired_cosine_distances(emb1, emb2))
    cosine_scores = cosine_scores * 5
    condition = cosine_scores < 0
    cosine_scores[condition] = 0

    cosine_scores = torch.FloatTensor(cosine_scores)
    labels = torch.FloatTensor(labels)
    pearson = torchmetrics.functional.pearson_corrcoef(cosine_scores, labels)
    
    if save_result:
        return pearson.item(), cosine_scores
    else:
        return pearson.item()
        
    
# 모델을 불러온 뒤 dev.csv 파일을 사용하여 점수를 계산하는 함수입니다.
def load_test(args, model_path):
    root_path = 'data/sbert/'
    model = SentenceTransformer.load(model_path)
    
    dev_df = pd.read_csv(args.dev_path)
    st1, st2, labels = convert2data(dev_df)
    evaluate(model, st1, st2, labels, batch_size=1)

# dev.scv 사용할 때 모든 데이터를 모델에게 한번에 줘야 해서 dataloder를 사용하지 않고 바로 tensor로 바꾸는 함수입니다.
def convert2data(df: pd.DataFrame):
    sentences1 = []
    sentences2 = []
    labels = []
    for i, data in tqdm(df.iterrows(), desc='SBERT inference_datas', total=len(df)):
        st1 = data['sentence_1']
        st2 = data['sentence_2']
        score = (data['label']) * 1.0
        sentences1.append(st1)
        sentences2.append(st2)
        labels.append(score)
    return sentences1, sentences2, labels
    
if __name__ == '__main__':
    args = get_args(mode="train")
    main(args)