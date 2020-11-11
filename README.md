# Fabric-Defect-Detector
## 1. 개요
- 목표
Detect defection of Fabric Database \
소량의 정상 섬유 이미지 데이터로 결함이 있는 섬유를 검출해낼 수 있는 딥러닝 학습 모델 제시.

- 프로젝트 소개
supervised learning을 위해 라벨링 작업을 하거나 임의로 결함 데이터를 생성하는 것은 비효율적이다. 
따라서, unsupervised learning을 통해 정상 데이터로만 결함을 검출하고자 한다.

  - Challenges
    - 데이터 양이 적다.
    - 정상 데이터만으로 모델을 학습시킨다.
  : 챌린지 극복을 위해 고화질의 이미지를 64 * 64 패치로 잘라내어 학습 데이터를 크게 증가시켰고, 정상 데이터만을 가지고 unsupervised learning model(Autoencoder)을 학습시키는 방법을 택했다.

## 2. 데이터
[AITEX사](https://www.aitex.es/afid/)에서 제공하는 섬유 이미지 데이터를 활용. \
- 원본 NoDefect(정상), Defect(결함)이미지를 구분해서 각각 디렉토리에 저장한다.
- getData.ipynb를 실행하면 train:val:test를 분리해서 각각 ./data/nocrop/의 train, val, val_for_training, test에 저장된다.
- getPatchImages.ipynb를 실행하면 이미지당 패치 이미지들이 ./patch/의 train, test에 저장된다.

📦Fabric-Defect-Detector
 ┣ 📂data
 ┃ ┣ 📂NODefect
 ┃ ┣ 📂Defect
 ┃ ┗ 📂nocrop
 ┃ ┃ ┣ 📂patch
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┣ 📂test
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┣ 📂val
 ┃ ┃ ┗ 📂val_for_training
 ┣ 📂model
 ┃ ┣ 📂finalepoch
 ┃ ┣ 📂model1
 ┃ ┣ 📂model2
 ┃ ┣ 📂model3
 ┃ ┣ 📂model4
 ┃ ┗ 📂model5
 ┣ 📜README.md
 ┣ 📜getData.ipynb
 ┣ 📜getPatchImages.ipynb
 ┣ 📜model-architecture.png
 ┣ 📜modeling.ipynb
 ┣ 📜project poster.pdf
 ┣ 📜test.ipynb
 ┗ 📜validation.ipynb

## 3. 모델 설명


## 4. 개발 환경
- Google Colab
[colab code](https://colab.research.google.com/drive/1-H9CfJZNQ8GDIg9eIgxabdGF-i_IitNH?usp=sharing)