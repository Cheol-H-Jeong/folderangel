"""Prompt templates for the planner stages.

Everything is in Korean by default because most target users are Korean; the
LLM easily handles English filenames inside a Korean prompt, so this is a safe
default that still generalises to mixed-language corpora.
"""
from __future__ import annotations

import json

STAGE_A_SYSTEM = """너는 수천 개의 파일을 의미적으로 잘 분류하는 전문 사서이다.
사용자가 지정한 폴더의 파일 목록과 요약을 받아 **카테고리 후보**를 제안한다.
원칙:
- 카테고리는 사람이 읽기 쉽도록 구체적이고 직관적인 한국어 이름을 사용한다.
- 지나치게 추상적인 이름(예: "기타", "문서", "자료")은 피한다.
- 확장자 기반의 기계적 분류(예: ".pdf 파일들")는 피하고 의미 기반 분류를 한다.
- 한 카테고리에 너무 적은 파일(1~2개)만 속하지 않도록 묶는다.
"""

STAGE_A_INSTRUCTION = """아래는 파일 배치이다. 이 배치에서 도출되는 카테고리 후보를 제안하라.
응답은 다음 JSON 스키마를 정확히 따라야 한다. 추가 텍스트 금지.
{
  "candidates": [
    {
      "id": "영문-소문자-슬러그",
      "name": "사람이 읽기 쉬운 한국어 폴더명",
      "description": "이 카테고리가 무엇을 담는지 한 줄 설명",
      "keywords": ["카테고리를 식별하는 키워드 3~6개"]
    }
  ]
}
"""

STAGE_MERGE_INSTRUCTION = """여러 배치에서 수집된 카테고리 후보들을 통합해서 최종 카테고리 목록을 만들어라.
- 비슷한 후보끼리 병합하고 이름을 정리하라.
- 최종 카테고리 수는 {min_categories}개 이상 {max_categories}개 이하로 한다.
- 너무 광범위해서 대부분 파일이 몰리는 단일 카테고리는 2개로 쪼개라.

응답 JSON 스키마(엄격):
{{
  "categories": [
    {{ "id": "slug", "name": "폴더명", "description": "설명" }}
  ]
}}
"""

STAGE_B_INSTRUCTION = """아래 파일 배치를 주어진 카테고리에만 분류하라.
- primary는 가장 적합한 단일 카테고리 id, primary_score는 0~1.
- primary_score와 비슷한(차이 ≤ {ambiguity_threshold}) 다른 카테고리들은 secondary 리스트에 score와 함께 포함한다.
- primary는 반드시 제공된 카테고리 id 중 하나여야 한다.
- reason은 왜 그렇게 분류했는지 한 줄(40자 이내) 한국어로 작성한다.

응답 JSON 스키마(엄격):
{{
  "assignments": [
    {{
      "path": "원본 파일 경로",
      "primary": "category_id",
      "primary_score": 0.0,
      "secondary": [ {{ "id": "category_id", "score": 0.0 }} ],
      "reason": "한 줄 사유"
    }}
  ]
}}
"""


def build_stage_a(files: list[dict]) -> str:
    body = json.dumps(files, ensure_ascii=False, indent=2)
    return f"{STAGE_A_SYSTEM}\n\n{STAGE_A_INSTRUCTION}\n\n파일 목록:\n{body}"


def build_stage_merge(candidate_sets: list[list[dict]], min_categories: int, max_categories: int) -> str:
    merged = {"batches": [{"candidates": cs} for cs in candidate_sets]}
    body = json.dumps(merged, ensure_ascii=False, indent=2)
    instr = STAGE_MERGE_INSTRUCTION.format(
        min_categories=min_categories, max_categories=max_categories
    )
    return f"{STAGE_A_SYSTEM}\n\n{instr}\n\n후보 묶음:\n{body}"


def build_stage_b(
    files: list[dict], categories: list[dict], ambiguity_threshold: float
) -> str:
    instr = STAGE_B_INSTRUCTION.format(ambiguity_threshold=ambiguity_threshold)
    body = json.dumps(
        {"categories": categories, "files": files}, ensure_ascii=False, indent=2
    )
    return f"{STAGE_A_SYSTEM}\n\n{instr}\n\n데이터:\n{body}"
