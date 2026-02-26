"""Korean morpheme feature extractor using kiwipiepy only."""
import json
import re
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from kiwipiepy import Kiwi

# ── Feature inventory ─────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent
_INVENTORY_PATH = _BASE_DIR / "feature_inventory_14-1.json"
FEATURE_INVENTORY: List[str] = json.loads(
    _INVENTORY_PATH.read_text(encoding="utf-8")
)["features"]

# ── POS tag → feature name ─────────────────────────────────────────────
# Covers all atomic features present in feature_inventory_14-1.json.
# kiwi-specific notes:
#   NNBC  단위성 의존 명사 (개, 명, …)
#   MMA/MMD/MMN  관형사 세부 분류 → 모두 "관형사"
#   VA-I / VV-I  불규칙 활용형 → 호출 전에 "-I" 제거
POS_TO_FEATURE: Dict[str, str] = {
    "NNG":  "일반 명사",
    "NNP":  "고유 명사",
    "NNB":  "의존 명사",
    "NNM":  "단위를 나타내는 명사",   # bareun 호환
    "NNBC": "단위를 나타내는 명사",   # kiwi 단위성 의존 명사
    "NP":   "대명사",
    "NR":   "수사",
    "VV":   "동사",
    "VA":   "형용사",
    "VX":   "보조 용언",
    "VCP":  "긍정 지정사",
    "VCN":  "부정 지정사",
    "MM":   "관형사",
    "MMA":  "관형사",
    "MMD":  "관형사",
    "MMN":  "관형사",
    "MAG":  "일반 부사",
    "MAJ":  "접속 부사",
    "JKS":  "주격 조사",
    "JKC":  "보격 조사",
    "JKG":  "관형격 조사",
    "JKO":  "목적격 조사",
    "JKB":  "부사격 조사",
    "JKV":  "호격 조사",
    "JKQ":  "인용격 조사",
    "JX":   "보조사",
    "JC":   "접속 조사",
    "EP":   "선어말 어미",
    "EF":   "종결 어미",
    "EC":   "연결 어미",
    "ETN":  "명사형 전성 어미",
    "ETM":  "관형형 전성 어미",
    "XPN":  "체언 접두사",
    "XSN":  "명사 파생 접미사",
    "XSV":  "동사 파생 접미사",
    "XSA":  "형용사 파생 접미사",
    "XR":   "어근",
    "SL":   "외국어",
    "SH":   "한자",
    "SN":   "숫자",
    "IC":   "감탄사",
    "UNA":  "UNA",
}

OPEN_BRACKETS  = {"(", "[", "{", "<", "«", "〈", "《", "「", "『", "【", "〔", "\u201c", "\u2018"}
CLOSE_BRACKETS = {")", "]", "}", ">", "»", "〉", "》", "」", "』", "】", "〕", "\u201d", "\u2019"}
HYPHEN_TOKENS  = {"-", "‐", "‑", "‒", "–", "—", "―"}
COUNTER_TOKENS = {
    "개", "명", "대", "마리", "권", "장", "줄", "병", "잔", "번", "회", "채", "층", "살",
    "킬로그램", "kg", "km", "cm", "m", "g", "리터", "L",
}

# ── Kiwi singleton ────────────────────────────────────────────────────
_KIWI: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


# ── Token → feature label ─────────────────────────────────────────────
def token_to_feature(token: str, pos: str) -> str:
    """Map a (surface, POS) pair to a feature name used in FEATURE_INVENTORY.

    feature_inventory_14-1 uses "마침표, 물음표, 느낌표" as a single combined
    feature for all sentence-ending punctuation (SF), while "줄임표" (…) is
    a separate feature.
    """
    if pos == "SF":
        if token in {"...", "…", "⋯"}:
            return "줄임표"
        return "마침표, 물음표, 느낌표"   # covers . ? !
    if token in HYPHEN_TOKENS or pos == "SO":
        return "붙임표"
    if pos == "SSO" or token in OPEN_BRACKETS:
        return "여는 괄호"
    if pos == "SSC" or token in CLOSE_BRACKETS:
        return "닫는 괄호"
    if pos in {"SC", "SP", "SE", "SS"}:
        return "구분자"
    return POS_TO_FEATURE.get(pos, "")


# ── N-gram count ──────────────────────────────────────────────────────
def count_ngrams(sequence: List[str], pattern: List[str]) -> int:
    n = len(pattern)
    if n == 0 or len(sequence) < n:
        return 0
    return sum(1 for i in range(len(sequence) - n + 1) if sequence[i: i + n] == pattern)


# ── Feature extraction from raw token list ────────────────────────────
def extract_features_from_raw_tokens(
    raw_tokens: List[Dict[str, str]],
    sentence_count: int | None = None,
) -> Dict[str, int]:
    """Build a {feature: count} dict for every entry in FEATURE_INVENTORY.

    raw_tokens: list of {"token": str, "rightPOS": str}
    """
    labels: List[str] = []
    counts: Counter = Counter()
    sent_end = 0

    token_pos_pairs = [
        (str(t.get("token", "")), str(t.get("rightPOS", "")))
        for t in raw_tokens
    ]
    for i, (token, pos) in enumerate(token_pos_pairs):
        feat = token_to_feature(token, pos)
        # Fallback heuristic: kiwi tags counter nouns as NNB; NNBC already
        # handled via POS_TO_FEATURE, but keep this for robustness.
        if not feat and pos in {"NNB", "NNG"} and token in COUNTER_TOKENS and i > 0:
            if token_pos_pairs[i - 1][1] in {"SN", "NR"}:
                feat = "단위를 나타내는 명사"
        if not feat:
            continue
        labels.append(feat)
        counts[feat] += 1
        if feat == "마침표, 물음표, 느낌표":
            sent_end += 1

    out: Dict[str, int] = {f: 0 for f in FEATURE_INVENTORY}
    for f in FEATURE_INVENTORY:
        if f == "문장수":
            if sentence_count is not None:
                out[f] = sentence_count
            else:
                out[f] = sent_end if sent_end > 0 else (1 if labels else 0)
        elif "+" in f:
            out[f] = count_ngrams(labels, [x.strip() for x in f.split("+")])
        else:
            out[f] = counts.get(f, 0)
    return out


# ── Stringify features for prompt ─────────────────────────────────────
def select_and_stringify(features: Dict[str, int]) -> str:
    nonzero = [{"name": k, "count": v} for k, v in features.items() if v > 0]
    if not nonzero:
        return "없음"
    if len(nonzero) > 3:
        nonzero = random.SystemRandom().sample(nonzero, 3)
    return ", ".join(f"{x['name']}: {x['count']}" for x in nonzero)


# ── Word grouping helper ───────────────────────────────────────────────
def _group_tokens_into_words(sent_text: str, tokens) -> List[Dict[str, Any]]:
    """Group kiwi morpheme tokens into whitespace-delimited 어절 words."""
    word_spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", sent_text)]
    words: List[Dict[str, Any]] = [{"surface": ws, "morphemes": []} for _, _, ws in word_spans]

    for token in tokens:
        tag = token.tag.split("-")[0] if "-" in str(token.tag) else str(token.tag)
        for i, (ws, we, _) in enumerate(word_spans):
            if ws <= token.start < we:
                words[i]["morphemes"].append({"surface": token.form, "tag": tag})
                break

    return [w for w in words if w["morphemes"]]


# ── Main analysis entry point ─────────────────────────────────────────
def analyze_with_kiwi(text: str) -> Dict[str, Any]:
    """Analyze Korean text with kiwipiepy.

    Returns:
      {
        "raw_tokens": [{"token": str, "rightPOS": str}, ...],
        "sentences": [
          {
            "text": str,
            "raw_token_count": int,
            "raw_tokens": [...],
            "words": [{"surface": str, "morphemes": [{"surface": str, "tag": str}]}, ...],
          },
          ...
        ],
      }
    """
    kiwi = _get_kiwi()
    normalized = (text or "").strip()
    if not normalized:
        return {"raw_tokens": [], "sentences": []}

    sents = kiwi.split_into_sents(normalized)

    raw_tokens: List[Dict[str, str]] = []
    sentences: List[Dict[str, Any]] = []

    for sent in sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        tokens = kiwi.tokenize(sent_text)

        sent_raw_tokens: List[Dict[str, str]] = []
        for token in tokens:
            tag = token.tag.split("-")[0] if "-" in str(token.tag) else str(token.tag)
            t = {"token": token.form, "rightPOS": tag}
            sent_raw_tokens.append(t)
            raw_tokens.append(t)

        words = _group_tokens_into_words(sent_text, tokens)

        sentences.append(
            {
                "text": sent_text,
                "raw_token_count": len(sent_raw_tokens),
                "raw_tokens": sent_raw_tokens,
                "words": words,
            }
        )

    return {"raw_tokens": raw_tokens, "sentences": sentences}


def main() -> None:
    parser = argparse.ArgumentParser(description="Get raw Kiwi POS tokens")
    parser.add_argument("--text", required=True, type=str)
    args = parser.parse_args()

    result = analyze_with_kiwi(args.text)
    output = {
        "backend": "kiwi",
        "sentence_count": len(result["sentences"]),
        "raw_token_count": len(result["raw_tokens"]),
        "raw_tokens": result["raw_tokens"],
        "sentences": result["sentences"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
