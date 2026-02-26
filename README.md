# Korean Essay Rater - Train

한국어 에세이 자동 채점 모델 학습 코드입니다.
EXAONE-3.5-7.8B 기반 QLoRA 파인튜닝으로, 8개 루브릭 항목(1~9점)에 대한 점수와 피드백을 생성합니다.

## 주요 특징

- **모델**: EXAONE-3.5-7.8B + QLoRA (r=32, alpha=64)
- **채점 항목**: 과제 수행의 충실성, 설명의 명료성/구체성/적절성, 문장의 연결성, 글의 통일성, 어휘의 적절성, 어법의 적절성
- **Loss**: EMO + Number Token Loss (MSE) + Class-Balanced Focal Loss, EMA 기반 동적 가중치
- **데이터 경량화**: 루브릭 분리 + 동적 Feature 샘플링으로 ~60% 토큰 절약
- **Feature 추출**: Kiwi 형태소 분석 기반 언어적 특징 추출 및 랜덤 샘플링

## 프로젝트 구조

```
├── run_train.sh            # 학습 실행 스크립트
├── train.py                # 메인 학습 코드
├── preprocess_data.py      # 데이터 전처리 (원본 → 경량화 JSONL)
├── data_precomputed/       # 전처리된 학습 데이터 (Git LFS)
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── modules/
    ├── aes_dataset.py              # 데이터셋 및 Collator
    ├── custom_trainer.py           # CustomTrainer (MTL loss)
    ├── prompt_template.py          # 프롬프트 템플릿 및 루브릭
    ├── feature_extractor.py        # Kiwi 기반 언어 특징 추출
    ├── feature_inventory_14-1.json # Feature 목록 정의
    ├── number_token_loss.py        # Number Token Loss
    ├── number_tokenizer.py         # 숫자 토크나이저
    ├── class_balanced_focal_loss.py # Class-Balanced Focal Loss
    ├── evaluate_module.py          # 평가 (QWK 등)
    └── inference_module.py         # 추론 모듈
```

## 실행 방법

### 요구사항

- Python 3.10+
- CUDA 지원 GPU (VRAM 24GB 이상 권장)
- EXAONE-3.5-7.8B 모델 가중치

### 주요 패키지

```
torch, transformers, peft, bitsandbytes, trl
scikit-learn, kiwipiepy, wandb
```

### 학습 실행

```bash
# 기본 실행
bash run_train.sh

# 옵션 예시
bash run_train.sh --dry_run          # 데이터/모델 로딩만 테스트
bash run_train.sh --no_wandb         # WandB 없이 학습
bash run_train.sh --epochs 4         # 에폭 수 변경
```

`run_train.sh` 내 `BASE_MODEL` 경로를 EXAONE 모델 위치에 맞게 수정하세요.

### 학습 설정 (기본값)

| 항목 | 값 |
|------|-----|
| Batch Size | 4 x 8 (effective 32) |
| Learning Rate | 2e-4 |
| Epochs | 8 |
| Max Seq Length | 1536 |
| Optimizer | paged_adamw_8bit |
| Sequence Packing | ON |
