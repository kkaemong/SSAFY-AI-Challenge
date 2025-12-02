## SSAFY-AI-Challenge  
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

- 문제는 **객관식 형태(VQA·Multimodal QA)**로 구성됨  
- 모델은 **이미지와 텍스트(문항 + 선택지)**를 동시에 입력받아  
  가장 적절한 선택지를 예측해야 함  
- 실제 시험 환경처럼  
  **시각적 단서 + 언어적 이해 + 문장 비교 능력**을 종합적으로 요구함  

본 실험에서는 Qwen2.5-VL 모델을 기반으로 LoRA 방식으로 파인튜닝하여  
다양한 유형의 시각 문제에 대해 높은 정확도를 달성하는 것을 목표로 진행했습니다.
