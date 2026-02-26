"""
Dynamic Feature Sampling AES Dataset & Collator
=================================================
학습-추론 Feature 불일치 해결:
- 학습 데이터에는 고정된 Feature가 라벨링되어 있지만
- 추론 시에는 kiwi로 동적 추출 후 랜덤 3개 샘플링
- 이를 해결하기 위해 학습 시에도 매 step마다 kiwi로 동적 Feature 추출/샘플링

Dataset은 경량화된 JSONL을 읽고, __getitem__ 호출 시마다:
1. kiwi로 에세이의 형태소 분석 수행
2. Feature inventory에서 비-zero Feature를 랜덤 3개 샘플링
3. 프롬프트 템플릿에 동적으로 주입하여 instruction 생성
"""

import json
import random
import unicodedata
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

from .prompt_template import build_instruction, build_output
from .feature_extractor import (
    analyze_with_kiwi,
    extract_features_from_raw_tokens,
)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class DynamicFeatureAESDataset(Dataset):
    """
    경량화된 JSONL 데이터를 로드하고, 매 접근 시 kiwi 동적 Feature 샘플링을 수행하는 Dataset.

    각 샘플은:
    - question, essay, keywords, scores, feedback 필드로 구성
    - __getitem__ 시 kiwi로 essay 분석 -> Feature 추출 -> 3개 랜덤 샘플링 -> 프롬프트 생성

    학습 시 DataLoader의 num_workers=0 으로 설정해야 kiwi 싱글턴이 정상 동작한다.
    (또는 worker별 kiwi 초기화가 필요)
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        num_features: int = 3,
        cache_features: bool = True,
        precomputed_features: bool = False,
    ):
        """
        Args:
            data_path: 경량화된 JSONL 파일 경로
            max_samples: 최대 샘플 수 (디버깅/소규모 실험용)
            num_features: 랜덤 샘플링할 Feature 수 (기본 3)
            cache_features: True면 kiwi 분석 결과를 캐시 (에포크 간 중복 분석 방지)
            precomputed_features: True면 JSONL의 features_dict 필드를 사용하여 kiwi 호출 생략
        """
        self.data: List[Dict[str, Any]] = []
        self.num_features = num_features
        self.cache_features = cache_features
        self.precomputed_features = precomputed_features
        self._feature_cache: Dict[int, Dict[str, int]] = {}

        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                self.data.append(sample)

                # precomputed features가 있으면 캐시에 직접 로드
                if precomputed_features and "features_dict" in sample:
                    self._feature_cache[i] = sample["features_dict"]

    def __len__(self) -> int:
        return len(self.data)

    def _extract_features(self, idx: int, essay: str) -> str:
        """Feature 추출 후 랜덤 3개 샘플링하여 문자열 반환.

        precomputed_features=True면 캐시에서 직접 로드 (kiwi 호출 없음).
        """
        if idx in self._feature_cache:
            features = self._feature_cache[idx]
        elif self.precomputed_features:
            # precomputed인데 캐시에 없으면 빈 dict (fallback)
            features = {}
        else:
            result = analyze_with_kiwi(essay)
            raw_tokens = result.get("raw_tokens", [])
            sent_count = len(result.get("sentences", []))
            features = extract_features_from_raw_tokens(
                raw_tokens, sentence_count=sent_count
            )
            if self.cache_features:
                self._feature_cache[idx] = features

        # 매번 랜덤 샘플링 (학습 step마다 다른 Feature 조합)
        return self._sample_features(features)

    def _sample_features(self, features: Dict[str, int]) -> str:
        """비-zero Feature에서 num_features개를 랜덤 샘플링."""
        nonzero = [{"name": k, "count": v} for k, v in features.items() if v > 0]
        if not nonzero:
            return "없음"
        if len(nonzero) > self.num_features:
            nonzero = random.sample(nonzero, self.num_features)
        return ", ".join(f"{x['name']}: {x['count']}" for x in nonzero)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        sample = self.data[idx]

        essay = sample["essay"]
        features_str = self._extract_features(idx, essay)

        instruction = build_instruction(
            question=sample["question"],
            essay=essay,
            keywords=sample["keywords"],
            features=features_str,
        )

        output = build_output(sample["scores"], sample.get("feedback", ""))

        return {
            "instruction": instruction,
            "output": output,
            "scores": sample["scores"],
        }


class AESCollatorMTL(DataCollatorForLanguageModeling):
    """
    MTL Collator: CE + NTL + EMO를 위한 다중 라벨 생성.

    - labels: CE 학습용 (instruction은 -100 마스킹, output 전체 학습)
    - ntl_labels: NTL 학습용 (output 중 '점수 파트(처음 score_token_len 토큰)'만)
    - emo_labels: EMO 학습용 (output 중 '피드백 파트'만)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 1536,
        score_token_len: int = 16,
        pad_to_max: bool = False,
        pack: bool = False,
    ):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eot_token = tokenizer.eos_token
        self.max_seq_length = int(max_seq_length)
        self.score_token_len = int(score_token_len)
        self.pad_to_max = bool(pad_to_max)
        self.pack = bool(pack)

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]):
        if self.pack:
            return self._collate_packed(examples)
        return self._collate_padded(examples)

    def _collate_padded(self, examples: List[Dict[str, Union[str, List[int]]]]):
        # 1) instruction + output + eos
        merged_sequences = [
            f"{normalize_text(ex['instruction'])}{normalize_text(ex['output'])}{self.eot_token}"
            for ex in examples
        ]

        padding_mode = "max_length" if self.pad_to_max else "longest"
        batch = self.tokenizer(
            merged_sequences,
            padding=padding_mode,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # 2) CE용 labels
        labels = batch["input_ids"].clone()

        q_lens = []
        for i, ex in enumerate(examples):
            q_ids = self.tokenizer(
                normalize_text(ex["instruction"]),
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )["input_ids"][0]
            q_len = int(q_ids.size(0))
            q_lens.append(q_len)
            labels[i, :q_len] = -100

        labels[batch["attention_mask"] == 0] = -100

        # 3) NTL용 ntl_labels: 점수 파트만
        ntl_labels = torch.full_like(labels, -100)
        emo_labels = torch.full_like(labels, -100)

        B, T = labels.shape
        for i in range(B):
            start = q_lens[i]
            end = min(start + self.score_token_len, T)
            if start < T and start < end:
                ntl_labels[i, start:end] = labels[i, start:end]
            if end < T:
                emo_labels[i, end:] = labels[i, end:]

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "ntl_labels": ntl_labels,
            "emo_labels": emo_labels,
        }

    def _collate_packed(self, examples: List[Dict[str, Union[str, List[int]]]]):
        """Sequence packing: 여러 짧은 샘플을 하나의 max_seq_length 시퀀스에 합쳐 패딩 낭비를 줄인다."""
        max_len = self.max_seq_length

        # 1) 각 샘플을 개별 토크나이즈하여 길이 측정
        sample_data = []
        for ex in examples:
            instr_text = normalize_text(ex["instruction"])
            merged_text = f"{instr_text}{normalize_text(ex['output'])}{self.eot_token}"

            merged_ids = self.tokenizer(
                merged_text, add_special_tokens=False,
                truncation=True, max_length=max_len,
            )["input_ids"]

            instr_ids = self.tokenizer(
                instr_text, add_special_tokens=False,
                truncation=True, max_length=max_len,
            )["input_ids"]

            sample_data.append({
                "input_ids": merged_ids,
                "q_len": len(instr_ids),
                "total_len": len(merged_ids),
            })

        # 2) First-fit-decreasing bin packing
        sorted_indices = sorted(
            range(len(sample_data)),
            key=lambda i: sample_data[i]["total_len"],
            reverse=True,
        )
        bins: List[List[int]] = []
        bin_remaining: List[int] = []

        for idx in sorted_indices:
            sample_len = sample_data[idx]["total_len"]
            placed = False
            for b in range(len(bins)):
                if bin_remaining[b] >= sample_len:
                    bins[b].append(idx)
                    bin_remaining[b] -= sample_len
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_remaining.append(max_len - sample_len)

        # 3) 각 bin에 대해 packed 텐서 생성
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_ntl_labels = []
        batch_emo_labels = []
        batch_position_ids = []

        for bin_samples in bins:
            packed_ids = []
            packed_labels = []
            packed_ntl = []
            packed_emo = []
            packed_pos = []

            for sample_idx in bin_samples:
                sd = sample_data[sample_idx]
                ids = sd["input_ids"]
                q_len = sd["q_len"]
                total_len = sd["total_len"]

                # input_ids
                packed_ids.extend(ids)

                # position_ids: 각 sub-sample마다 0부터 리셋
                packed_pos.extend(list(range(total_len)))

                # labels (CE): instruction 부분은 -100
                sample_labels = [-100] * q_len + list(ids[q_len:])
                packed_labels.extend(sample_labels)

                # ntl_labels: score 토큰만
                sample_ntl = [-100] * total_len
                score_end = min(q_len + self.score_token_len, total_len)
                for j in range(q_len, score_end):
                    sample_ntl[j] = ids[j]
                packed_ntl.extend(sample_ntl)

                # emo_labels: feedback 토큰만
                sample_emo = [-100] * total_len
                for j in range(score_end, total_len):
                    sample_emo[j] = ids[j]
                packed_emo.extend(sample_emo)

            # 패딩
            current_len = len(packed_ids)
            pad_len = max_len - current_len

            packed_ids.extend([self.pad_token_id] * pad_len)
            packed_labels.extend([-100] * pad_len)
            packed_ntl.extend([-100] * pad_len)
            packed_emo.extend([-100] * pad_len)
            packed_pos.extend([0] * pad_len)
            attn_mask = [1] * current_len + [0] * pad_len

            batch_input_ids.append(packed_ids)
            batch_attention_mask.append(attn_mask)
            batch_labels.append(packed_labels)
            batch_ntl_labels.append(packed_ntl)
            batch_emo_labels.append(packed_emo)
            batch_position_ids.append(packed_pos)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(batch_position_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "ntl_labels": torch.tensor(batch_ntl_labels, dtype=torch.long),
            "emo_labels": torch.tensor(batch_emo_labels, dtype=torch.long),
        }
