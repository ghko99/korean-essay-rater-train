"""
데이터셋 경량화 전처리 스크립트
================================
기존 aes_dataset_mtl의 instruction 필드에 루브릭, 에세이, 키워드, 특징이 모두 하나의
긴 텍스트로 합쳐져 있어 토큰 수가 불필요하게 많다.

이 스크립트는:
1. 원본 JSONL에서 구조화된 필드(question, essay, keywords, features)를 파싱
2. 루브릭을 instruction에서 제거하고 학습 시 프롬프트 템플릿에서 주입 (공유 상수)
3. output에서 scores와 feedback을 분리
4. 경량화된 새 JSONL을 생성

결과적으로 instruction 필드 크기가 ~60% 줄어들어 학습 속도가 대폭 향상된다.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional

from modules.feature_extractor import (
    analyze_with_kiwi,
    extract_features_from_raw_tokens,
)


def parse_instruction(instruction: str) -> Dict[str, str]:
    """기존 instruction 텍스트에서 구조화된 필드를 추출."""
    result = {
        "question": "",
        "essay": "",
        "keywords": "",
        "features": "",
    }

    # 에세이 질문 추출
    m = re.search(r"### 에세이 질문:\s*\n(.+?)(?=\n### )", instruction, re.DOTALL)
    if m:
        result["question"] = m.group(1).strip()

    # 학생 에세이 추출
    m = re.search(r"### 학생 에세이:\s*\n(.+?)(?=\n### )", instruction, re.DOTALL)
    if m:
        result["essay"] = m.group(1).strip()

    # 핵심 키워드 추출
    m = re.search(r"- 핵심 키워드:\s*(.+)", instruction)
    if m:
        result["keywords"] = m.group(1).strip()

    # 주요 언어적 특징 추출 (학습 시 동적 샘플링으로 대체되므로 참고용)
    m = re.search(r"- 주요 언어적 특징:\s*(.+)", instruction)
    if m:
        result["features"] = m.group(1).strip()

    return result


def parse_output(output: str) -> Dict[str, str]:
    """output에서 scores 줄과 feedback을 분리."""
    lines = output.strip().split("\n")
    scores_line = lines[0].strip() if lines else ""

    # feedback 부분 추출
    feedback = ""
    feedback_start = output.find("### Feedback:")
    if feedback_start >= 0:
        feedback = output[feedback_start:].strip()
    elif len(lines) > 2:
        # ### Feedback: 헤더가 없는 경우 두 번째 줄부터
        feedback = "\n".join(lines[2:]).strip()

    return {"scores": scores_line, "feedback": feedback}


def process_sample(sample: Dict) -> Optional[Dict]:
    """단일 샘플을 경량화된 형식으로 변환."""
    # 이미 경량 포맷(question/essay/scores)인 경우도 그대로 통과
    if "question" in sample and "essay" in sample and "scores" in sample:
        scores = sample.get("scores", [])
        if not isinstance(scores, list) or len(scores) < 8:
            return None
        scores = [int(s) for s in scores[:8]]
        if not all(1 <= s <= 9 for s in scores):
            return None

        result = {
            "question": sample.get("question", "").strip(),
            "essay": sample.get("essay", "").strip(),
            "keywords": sample.get("keywords", "").strip(),
            "scores": scores,
            "feedback": sample.get("feedback", ""),
        }
        if not result["question"] or not result["essay"]:
            return None
        if "grader_1_scores" in sample:
            result["grader_1_scores"] = sample["grader_1_scores"]
        if "grader_2_scores" in sample:
            result["grader_2_scores"] = sample["grader_2_scores"]
        return result

    instruction = sample.get("instruction", "")
    output = sample.get("output", "")

    parsed_inst = parse_instruction(instruction)
    parsed_out = parse_output(output)

    # 필수 필드 검증
    if not parsed_inst["question"] or not parsed_inst["essay"]:
        return None

    # scores 검증 (8개의 1-9 숫자)
    score_nums = re.findall(r"\d+", parsed_out["scores"])
    if len(score_nums) < 8:
        return None

    scores = [int(n) for n in score_nums[:8]]
    if not all(1 <= s <= 9 for s in scores):
        return None

    result = {
        "question": parsed_inst["question"],
        "essay": parsed_inst["essay"],
        "keywords": parsed_inst["keywords"],
        "scores": scores,
        "feedback": parsed_out["feedback"],
    }

    # grader scores 보존 (존재하면)
    if "grader_1_scores" in sample:
        result["grader_1_scores"] = sample["grader_1_scores"]
    if "grader_2_scores" in sample:
        result["grader_2_scores"] = sample["grader_2_scores"]

    return result


def extract_features_for_essay(essay: str) -> Dict[str, int]:
    """kiwipiepy로 에세이 형태소 분석 후 feature dict 추출."""
    result = analyze_with_kiwi(essay)
    raw_tokens = result.get("raw_tokens", [])
    sent_count = len(result.get("sentences", []))
    features = extract_features_from_raw_tokens(raw_tokens, sentence_count=sent_count)
    # non-zero features만 저장하여 JSONL 크기 절약
    return {k: v for k, v in features.items() if v > 0}


def process_file(input_path: str, output_path: str, extract_features: bool = False) -> int:
    """JSONL 파일을 처리하여 경량화된 JSONL 생성."""
    processed = 0
    skipped = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                sample = json.loads(line.strip())
                result = process_sample(sample)
                if result:
                    if extract_features:
                        features_dict = extract_features_for_essay(result["essay"])
                        result["features_dict"] = features_dict
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed += 1
                else:
                    skipped += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Line {line_num}: {e}")
                skipped += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="데이터셋 경량화 전처리")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/khko/workspace/Automated_Essay_Scoring/aes_datasets/aes_dataset_mtl",
        help="원본 데이터셋 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="경량화된 데이터셋 출력 디렉토리",
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="kiwipiepy로 형태소 분석 Feature를 사전 추출하여 features_dict 필드에 저장",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        input_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.jsonl"

        if not input_path.exists():
            print(f"[SKIP] {input_path} not found")
            continue

        print(f"Processing {split}..." + (" (with feature extraction)" if args.extract_features else ""))
        count = process_file(str(input_path), str(output_path), extract_features=args.extract_features)
        print(f"  -> {count} samples written to {output_path}")

    # 원본 vs 경량화 크기 비교
    print("\n=== Size Comparison ===")
    for split in ["train", "valid", "test"]:
        orig = input_dir / f"{split}.jsonl"
        new = output_dir / f"{split}.jsonl"
        if orig.exists() and new.exists():
            orig_size = orig.stat().st_size / (1024 * 1024)
            new_size = new.stat().st_size / (1024 * 1024)
            ratio = (1 - new_size / orig_size) * 100
            print(f"  {split}: {orig_size:.1f}MB -> {new_size:.1f}MB ({ratio:.1f}% reduction)")


if __name__ == "__main__":
    main()
