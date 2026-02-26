"""
프롬프트 템플릿 및 루브릭 상수
==============================
루브릭은 모든 샘플에서 동일하므로 데이터에서 제거하고 여기서 주입한다.
이를 통해 토큰 수를 ~400 토큰 절약할 수 있다.
"""

# 루브릭 (9점 최고, 1점 최저) - 간결화된 버전
RUBRIC_TEXT = """- 과제 수행의 충실성: 9점(완벽한 과제 수행) ~ 1점(매우 미흡)
- 설명의 명료성: 9점(대상에 초점 맞춘 완벽한 서술) ~ 1점(매우 미흡)
- 설명의 구체성: 9점(구체적 세부 내용 제시) ~ 1점(세부 내용 없음)
- 설명의 적절성: 9점(핵심어 4개 이상 활용한 내용 구성) ~ 1점(핵심어 없음)
- 문장의 연결성: 9점(오류 없이 문장간 연결 우수) ~ 1점(연결 매우 부족)
- 글의 통일성: 9점(통일성을 잘 갖춤) ~ 1점(통일성 매우 미흡)
- 어휘의 적절성: 9점(어휘 탁월, 문장 수려) ~ 1점(어휘 부적절, 표현 어색)
- 어법의 적절성: 9점(맞춤법/띄어쓰기 완벽) ~ 1점(전반적으로 미흡)"""

RUBRIC_CRITERIA = [
    "과제 수행의 충실성",
    "설명의 명료성",
    "설명의 구체성",
    "설명의 적절성",
    "문장의 연결성",
    "글의 통일성",
    "어휘의 적절성",
    "어법의 적절성",
]

# 프롬프트 템플릿 - 경량화: 루브릭을 간결하게, 불필요한 줄바꿈 제거
PROMPT_TEMPLATE = """### 지시문:
너는 루브릭에 따라 학생 에세이를 평가하는 AI 채점기다.
### 에세이 질문:
{question}
### 학생 에세이:
{essay}
### 관련 정보:
- 핵심 키워드: {keywords}
- 주요 언어적 특징: {features}
### 채점 기준 (루브릭):
{rubric}

8개의 점수를 공백으로 구분해 한 줄에 출력하고, 빈 줄 이후에 각 항목에 대한 상세 피드백을 작성하라.
"""


def build_instruction(
    question: str,
    essay: str,
    keywords: str,
    features: str,
) -> str:
    """동적으로 프롬프트를 구성."""
    return PROMPT_TEMPLATE.format(
        question=question,
        essay=essay,
        keywords=keywords,
        features=features,
        rubric=RUBRIC_TEXT,
    )


def build_output(scores: list, feedback: str) -> str:
    """scores + feedback을 결합하여 output 문자열 생성."""
    scores_str = " ".join(str(s) for s in scores)
    if feedback:
        return f"{scores_str}\n\n{feedback}"
    return scores_str
