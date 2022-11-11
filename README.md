

# 🌠 STS - IDLE 조 level 1 대회 Project
---
**프로젝트 주제 : 문맥적 유사도 측정**

**프로젝트 개요**  
- 두 문장의 문맥적 의미의 유사성을 판단하여 0.0~5.0 사이의 점수로 출력하는 모델을 설계하고 개선 방법론을 적용하여 성능을 개선하고자 함 
- 현 시점까지 부스트캠프에서 배운 Transformer 기반의 사전 학습 모델, 학습방법, 데이터 전처리 등을 종합적으로 시도해보고자 하였음

**활용 장비 및 재료**
- GPU : v100 * 5
- 협업 툴 : Github, Notion, Wandb
- 개발 환경 : Ubuntu 18.04
- 라이브러리 : torch 1.12.0, torchmetrics 0.10.0, wandb 0.13.4, sentence-transformers 2.2.2

### Command Line Interface

##### Train phase
```
>>> cd code
>>> python train.py 
	--model_name=[model name] 
	--version=[model 버전 명] 
	--usd_dev=[Boolean]
	--clean=[Boolean]
```

##### Inference phase
```
>>> cd code
>>> python inference.py
	 --model_name=[model name] 
	 --version=[model 버전 명] 
	 --checkpoint_path=[ckpt 폴더 내 선택할 경로명]
```

### Project Directories
```
├─ code
│  ├─ args.py
│  ├─ inference.py
│  ├─ requirements.txt
│  ├─ sts
│  │  ├─ dataloader.py
│  │  ├─ metric.py
│  │  ├─ model.py
│  │  └─ utils.py
│  └─ train.py
├─ notebooks
│ └─ EDA.ipynb
└─ data
   ├─ saved_models
   ├─ submissions
   └─ wandb_checkpoints
```

### Memebers 👥
---
**공통** : hyperparameter 조정 및 실험
-   **김지수**(팀장) [Github](https://github.com/kuotient)
    - Project Manager, 프로젝트 전체 구성, 학습 개선 팀
        -   프로젝트 리드
        -   학습 개선: K-fold, 학습 분석 툴 wandb, 코드 리팩토링
-   **김산**(팀원) [Github](https://github.com/jtlsan)
    - 데이터 팀, 모델 팀
        -   특수문자 제거 기능 구현
        -   SBERT 학습 구현
-   **박수현**(팀원) [Github](https://github.com/HitHereX)
    - 데이터 팀
        -   주어진 데이터에 대한 EDA 분석
        -   retranslation을 통한 데이터 증강 및 증강 데이터 EDA
-   **엄주언**(팀원) [Github](https://github.com/EJueon)
    - 모델 팀, 학습 개선 팀
        -   데이터 : 주어진 데이터에 대한 EDA 분석
        -   모델 : Loss Function 변경 및 추가 , 모델 partial freezing
        -   학습 개선 : Ensemble voting, stacking, Stratified K-fold
-   **현승엽**(팀원) Github
    - 모델 팀
        -   Baseline 모델에 linear layer를 추가한 모형 구현
