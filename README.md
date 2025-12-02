# SSAFY-AI-Challenge  
이미지 기반 4지선다형 문제 해결을 위한 AI 모델 개발 프로젝트

<p align="center">
  <img src="https://github.com/user-attachments/assets/ddf1eae8-9476-47c2-8d5d-17a4e14a4487" width="30%" />
  <img src="https://github.com/user-attachments/assets/7147d1d8-0c4c-412d-bd87-2743ef41d954" width="30%" />
  <img src="https://github.com/user-attachments/assets/a4e918b1-d180-424b-bca2-ad02d99aa4e1" width="30%" />
</p>

---

## 📌 프로젝트 개요

본 프로젝트는 **제공된 이미지와 4개의 선택지(문자열 선택지 포함)**를 기반으로  
모델이 올바른 정답을 선택하도록 학습시키는 **이미지 기반 4지선다형 문제 해결 AI 모델 개발 프로젝트**입니다.  
모델은 이미지의 시각적 특징과 텍스트 선택지를 동시에 이해하고, 최적의 정답을 예측하도록 학습되었습니다.

---

## 🎯 프로젝트 목표

- 이미지 + 선택지 조합을 이해하는 **멀티모달 모델 성능 최적화**
- 짧은 개발 기간 내에서 **최대한 높은 정확도 달성**
- 데이터 전처리 → 모델 학습 → 추론 → 제출 파일 생성까지  
  **End-to-End 파이프라인 완성**
- Kaggle 제출 규칙에 맞는 **정답 예측 CSV 파일 생성 및 제출 자동화**

---

## 📆 프로젝트 기간  
**2025.10.23 ~ 2025.10.27 (총 4일)**

짧은 기간 동안 이미지 기반 4지선다형 문제를 해결하는 AI 모델을 구축하고,  
제출 파일(csv)을 생성하여 Kaggle 평가 시스템에서 정확도를 겨루는 챌린지 형태로 진행되었습니다.

---

## 🏆 평가 기준 (Kaggle)

프로젝트 성능 평가는 **Kaggle 리더보드**에서 진행되며  
모델이 생성한 예측 정답을 아래 형식의 CSV로 제출하여 정확도를 평가했습니다.
정확도(Accuracy)를 기준으로 모델 성능이 결정됩니다.

---

## 🧠 사용 기술 (AI 중심)

- **Qwen2.5-VL**  
  멀티모달(이미지+텍스트) 문제 해결을 위한 비전-언어 모델

- **HuggingFace Transformers**  
  모델 로딩, 토크나이징, Chat Template 구성, 추론 파이프라인 구현

- **LoRA / PEFT**  
  대형 모델을 경량으로 학습시키기 위한 파라미터 효율적 미세조정 기법

- **BitsAndBytes (4bit / 8bit Quantization)**  
  GPU 메모리 절감을 위한 양자화 기술

- **Mixed Precision Training (bfloat16 / float16)**  
  AMP 기반 학습 속도 향상 및 메모리 최적화  
  → 특히 `bfloat16`을 활용해 안정적인 학습 환경 구축

- **PyTorch**  
  Optimizer, Scheduler, AMP, DataLoader 등 전체 학습 엔진 구성

---

## 🔧 핵심 코드 개선 사항 (모델 정확도 향상 중심)

짧은 기간 동안 높은 정확도를 달성하기 위해 학습 코드를 지속적으로 개선하며 실험을 반복했습니다.  
아래는 실제로 적용한 핵심 개선 포인트들입니다.

---

### 1️⃣ Optimizer 변경 (AdamW8bit → AdamW)

**문제**  
- bitsandbytes `AdamW8bit` 사용 시 충돌 및 메모리 불안정 발생  

**개선**

```python
# 변경 전
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)

# 변경 후
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
````

---

### 2️⃣ 혼합 정밀도 변경 (fp16 → bf16)

**문제**

* fp16에서 gradient overflow, loss nan 문제 발생

**개선**

```python
# 변경 전
with torch.cuda.amp.autocast(dtype=torch.float16):

# 변경 후
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
```

---

### 3️⃣ Gradient Accumulation 조정 (8 → 4)

**문제**

* Accumulation=8은 안정적이나 학습 속도가 느림

**개선**

```python
# 변경 전
GRAD_ACCUM = 8

# 변경 후
GRAD_ACCUM = 4
```

---

### 4️⃣ Warmup 비율 조정 (10% → 3%)

**문제**

* 4일짜리 대회에서 warmup 10%는 비효율적

**개선**

```python
# 변경 전
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps * 0.1), num_training_steps
)

# 변경 후
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps * 0.03), num_training_steps
)
```

---

### 5️⃣ Epoch 감소 (2 → 1)

**문제**

* GPU 시간 부족, 제출 사이클 확보 필요

**개선**

```python
# 변경 전
for epoch in range(2):

# 변경 후
for epoch in range(1):
```

---

### 6️⃣ DataLoader 설정 안정화

**문제**

* batch_size=2 → OOM 발생
* num_workers=2 → 멀티프로세싱 충돌 발생

**개선**

```python
# 변경 전
train_loader = DataLoader(train_ds, batch_size=2, num_workers=2)

# 변경 후
train_loader = DataLoader(train_ds, batch_size=1, num_workers=0)
```

---

### 7️⃣ AMP + GradScaler 안정적 유지

```python
# 변경 전
scaler = torch.cuda.amp.GradScaler(enabled=False)

# 변경 후
scaler = torch.cuda.amp.GradScaler(enabled=True)
```
## 📈 최종 모델 성능

본 프로젝트는 짧은 기간 동안 모델 구조, 학습 파이프라인, 하이퍼파라미터를 지속적으로 수정하며  
기존 정확도 **0.75**에서 최종적으로 **0.81**까지 성능을 향상시키는 데 성공하였습니다.
worker 수·batch size 조정)로 환경 충돌 제거

### 🏁 최종 결과
- **Initial Accuracy:** 0.75  
- **Final Accuracy:** 0.81  
- **Accuracy Improvement:** +0.06

짧은 4일간의 대회 기간 동안 모델 구조와 학습 최적화 전략을 조합하여  
실제 리더보드 기준으로 의미 있는 성능 향상을 달성하였습니다.

<img width="1070" height="71" alt="스크린샷 2025-10-26 오후 7 46 37" src="https://github.com/user-attachments/assets/0bfa916b-c680-4c27-90bd-93eed793dcfd" />
