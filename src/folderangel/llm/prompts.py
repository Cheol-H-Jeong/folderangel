"""Prompt templates for the planner stages.

Everything is in Korean by default because most target users are Korean; the
LLM easily handles English filenames inside a Korean prompt, so this is a safe
default that still generalises to mixed-language corpora.
"""
from __future__ import annotations

import json

STAGE_A_SYSTEM = """너는 회사/기관의 **프로젝트/사업 단위 폴더 정리 전문가**이다.
업무 파일은 보통 같은 사업·프로젝트·고객사 이름이 파일명에 반복적으로 등장한다.
너의 임무는 그 **공통된 프로젝트명, 사업명, 기관/고객사 이름, 제품/시스템 이름**을 식별해서
같은 프로젝트의 파일들이 한 폴더에 모이도록 폴더 체계를 설계하는 것이다.

원칙:
- 카테고리는 가능한 한 **구체적인 프로젝트명/사업명/고객사명**으로 만든다.
  좋은 예: "한국지역정보개발원 초거대 AI 공통기반", "AVOCA 특허 명세서", "사숲 챗봇 RAG"
  피할 예: "문서", "보고서", "프레젠테이션", "업무 자료", "기타", "회사 문서"
- 파일명·본문에서 반복되는 **고유명사·약어·버전 패턴**(예: v1.0, R1, 240301)을 단서로 그룹을 묶는다.
  같은 사업의 v0.5, R1, 최종_4 같은 버전 파일들은 한 폴더로 묶는다.
- 단발성 잡파일은 "프로젝트 외 자료" 같은 하나의 카테고리로 모아도 된다.
  단, 식별 가능한 프로젝트가 보이면 **프로젝트 폴더를 더 우선**으로 만든다.
- 확장자 기반 분류(예: "PPTX 파일들")는 절대 하지 않는다.
- 폴더명은 사람이 한눈에 어떤 프로젝트인지 알 수 있게 한국어(필요 시 영어 약어 포함)로 작성한다.
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


SINGLE_CALL_INSTRUCTION = """아래 전체 파일 목록을 보고, **한 번의 응답으로**
1) 프로젝트/사업/고객사 단위의 폴더 체계({min_categories}~{max_categories}개)를 결정하고,
2) 각 파일을 그 폴더에 분류하라.

요구사항:
- categories[].name 은 가능한 한 **구체적인 프로젝트/사업/고객사 이름**.
- assignments 의 path 는 입력으로 준 path 값을 **그대로** 복사한다(요약·축약·번역 금지).
- primary 는 categories[].id 중 하나여야 한다.
- primary_score 와 차이가 {ambiguity_threshold} 이하인 다른 후보가 있으면 secondary 에 넣는다.
- reason 은 어떤 프로젝트/단서로 묶었는지 한 줄(40자 이내).

응답 JSON 스키마(엄격, 추가 텍스트 금지):
{{
  "categories": [
    {{ "id": "kebab-slug", "name": "구체 프로젝트/사업명", "description": "한 줄 설명" }}
  ],
  "assignments": [
    {{
      "path": "<입력 path 그대로>",
      "primary": "<categories[].id>",
      "primary_score": 0.0,
      "secondary": [ {{ "id": "category_id", "score": 0.0 }} ],
      "reason": "한 줄 사유"
    }}
  ]
}}
"""


def build_single_call(
    files: list[dict],
    min_categories: int,
    max_categories: int,
    ambiguity_threshold: float,
) -> str:
    instr = SINGLE_CALL_INSTRUCTION.format(
        min_categories=min_categories,
        max_categories=max_categories,
        ambiguity_threshold=ambiguity_threshold,
    )
    body = json.dumps({"files": files}, ensure_ascii=False, indent=2)
    return f"{STAGE_A_SYSTEM}\n\n{instr}\n\n데이터:\n{body}"


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
