# Uni-d 해커톤 - 누수 유형 분류 프로젝트

## 📌 프로젝트 개요

이 프로젝트는 Uni-d 해커톤에서 진행된 **누수 유형 분류(Leak Type Classification)** 머신러닝 프로젝트입니다. 센서 데이터(C01~C26)를 활용하여 5가지 누수 유형을 분류하는 것이 목표입니다.

## 🎯 분류 목표

다음 5가지 누수 유형을 분류합니다:
- **0**: out (외부 누수)
- **1**: in (내부 누수)
- **2**: noise (노이즈)
- **3**: other (기타)
- **4**: normal (정상)

## 📊 데이터셋

### 데이터 구조
- **Training set**: 62,564개 샘플
- **Validation set**: 7,820개 샘플
- **Test set**: 7,820개 샘플

### 특징(Features)
- **site**: 사이트 ID
- **sid**: 센서 ID
- **C01 ~ C26**: 26개의 센서 측정값
- **leaktype**: 누수 유형 레이블 (타겟 변수)

## 🛠️ 기술 스택

### 주요 라이브러리
- **PyTorch**: 딥러닝 모델 구현
- **XGBoost**: 앙상블 학습
- **scikit-learn**: 전처리 및 평가
- **pandas**: 데이터 처리
- **matplotlib**: 시각화

## 🔬 모델 아키텍처

### 1. Neural Network (PyTorch)

#### 모델 구조
```python
Input (26) → FC (256) → ReLU 
           → FC (256) → ReLU 
           → FC (32) → ReLU 
           → FC (32) → ReLU 
           → FC (5) → ReLU
```

#### 하이퍼파라미터
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs**: 500

#### 성능
- **Final Training Loss**: 0.443
- **Final Validation F1 Score**: 0.697

### 2. XGBoost Regressor

#### 최적 하이퍼파라미터 (GridSearchCV)
- **learning_rate**: 0.5
- **max_depth**: 6
- **n_estimators**: 100
- **subsample**: 1

#### 성능
- **Validation F1 Score**: 0.735 ✨

## 🚀 사용 방법

### 1. 환경 설정
```bash
pip install torch pandas numpy scikit-learn xgboost matplotlib lightgbm
```

### 2. 데이터 준비
Google Drive에 다음과 같은 구조로 데이터를 배치:
```
/content/drive/MyDrive/
├── data/
│   ├── train/
│   ├── val/
│   └── dataset/
│       └── test.csv
└── Uni-d 데이터톤/
    └── data/
        └── sample_submission.csv
```

### 3. 모델 학습 및 추론

#### Neural Network
```python
# 모델 초기화
model = MyNeuralNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

# 학습
train(num_epochs=500)

# 추론
final_model = MyNeuralNet()
final_model.load_state_dict(torch.load('newModel.pth'))
predictions = torch.argmax(final_model(X_test), dim=1)
```

#### XGBoost
```python
# 모델 학습
xgb_model = xgb.XGBRegressor(
    learning_rate=0.5,
    max_depth=6,
    n_estimators=100,
    objective='multi:softprob'
)
xgb_model.fit(train_x, train_y)

# 예측
y_pred = xgb_model.predict(test_x).argmax(axis=1)
```

## 📈 전처리 과정

1. **데이터 로딩**: CSV 파일에서 train/val/test 데이터 로드
2. **컬럼 제거**: 'site', 'sid' 컬럼 제거
3. **레이블 인코딩**: 문자열 레이블을 숫자로 변환
4. **Z-Score 정규화**: 각 데이터셋에 대해 독립적으로 정규화 수행
5. **텐서 변환**: NumPy 배열을 PyTorch 텐서로 변환

## 📊 결과 분석

### 학습 곡선
프로젝트에서는 다음 시각화를 제공합니다:
- Training Loss curve (500 epochs)
- Validation F1 Score curve (500 epochs)
- XGBoost Feature Importance
- XGBoost Decision Tree 시각화

### 모델 비교
| 모델 | Validation F1 Score |
|------|---------------------|
| Neural Network | 0.697 |
| **XGBoost** | **0.735** ⭐ |

XGBoost 모델이 더 높은 성능을 보였습니다.

## 📝 주요 파일

- **result.ipynb**: 전체 실험 노트북 (1,718 lines)
  - Neural Network 구현 및 학습
  - XGBoost 모델 학습 및 하이퍼파라미터 튜닝
  - 결과 시각화 및 제출 파일 생성

## 🔑 핵심 기술 포인트

1. **멀티 모델 앙상블 접근**: Neural Network와 XGBoost 두 가지 방법론 비교
2. **GridSearchCV**: 체계적인 하이퍼파라미터 튜닝
3. **F1 Score 최적화**: 불균형 데이터셋에 적합한 평가 지표 사용
4. **조기 종료**: Validation F1 Score 기반 최적 모델 저장

## 📌 개선 가능한 부분

- [ ] Ensemble 방법론 적용 (Neural Network + XGBoost)
- [ ] 추가 특징 공학 (Feature Engineering)
- [ ] Cross-validation 적용
- [ ] 데이터 증강 기법 적용
- [ ] 클래스 불균형 처리 (SMOTE 등)

## 👤 개발자

- **GitHub**: [@eojin16](https://github.com/eojin16)
- **Last Update**: November 13, 2022

## 📄 라이선스

이 프로젝트는 Uni-d 해커톤의 일환으로 진행되었습니다.

---

**Note**: 이 프로젝트는 Google Colab 환경에서 실행되도록 설계되었습니다. Google Drive 마운트가 필요합니다.
