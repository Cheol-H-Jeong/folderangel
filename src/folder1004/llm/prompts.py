"""Prompt templates for the planner stages.

Everything is in Korean by default because most target users are Korean; the
LLM easily handles English filenames inside a Korean prompt, so this is a safe
default that still generalises to mixed-language corpora.
"""
from __future__ import annotations

import json


def _guidance_block(classification_guidance: str = "") -> str:
    text = (classification_guidance or "").strip()
    if not text:
        return ""
    text = text[:2000]
    return (
        "\n\n사용자가 이번 정리에 대해 추가로 지정한 분류 원칙이다. "
        "아래 원칙은 시스템 원칙과 충돌하지 않는 범위에서 강하게 반영하라.\n"
        "--- 사용자 분류 원칙 ---\n"
        f"{text}\n"
        "--- 끝 ---"
    )

STAGE_A_SYSTEM = """너는 회사/기관의 **프로젝트/사업 + 목적·용도 이중 축 폴더 정리 전문가**이다.
업무 파일은 보통 같은 사업·프로젝트·고객사 이름이 파일명에 반복적으로 등장한다.
너의 임무는 (1) 공통된 프로젝트명/사업명/기관/고객사/시스템 이름을 식별하고,
(2) 동시에 **목적·용도(산출물 / 구매·계약 / 회계·세금계산서 / 회의록 /
미디어·이미지·영상 자료 / 참고 자료)** 축으로도 분리해 같은 사업의 파일이라도
용도가 다르면 별도 폴더에 모이도록 폴더 체계를 설계하는 것이다.

**중요 — 단순히 사업명이 같다고 한 폴더로 합치지 마라.**
한 폴더는 *한 사업 × 한 용도* 의 교집합이어야 한다.  같은 사업에서
발생했더라도 다음과 같이 *기능이 다른* 파일들은 별도 폴더로 분리한다:
  · *작업물·산출물* — 사업의 주된 결과물
  · *거래·계약*     — 외부와의 약정·발주·매매·정산을 입증하는 문서
  · *회계·금융*     — 세무·정산·비용 처리·증빙 등 금전 흐름 기록
  · *회의·일정*     — 회의록·공지·일정·내부 커뮤니케이션
  · *원자료·미디어* — 본문 파싱이 사실상 불가능한 사진·영상·녹음·스캔 등
  · *외부 참고*     — 외부 보고서·표준안·논문 등 코퍼스 외부에서 들여온 자료
이 6개 축은 *모든 도메인*에 적용되는 일반 분류이다.  각 축에 어떤 파일이
속하는지는 너가 직접 판단한다.  내가 키워드 목록을 주지 않은 것은 그
판단을 너에게 맡긴 것이다 — 파일명·본문의 단서로 *기능적 의도*를 읽어라.

판정의 우선 순위:
  1) 사업/프로젝트 식별이 가능한가? → 사업 축이 결정된다.
  2) 그 안에서 이 파일은 어떤 *기능*을 수행하는가? → 용도 축이 결정된다.
  3) 사업 식별은 안 되지만 용도가 분명하면 *용도-단독 폴더* 로 보낸다
     (예: 코퍼스 전체의 회계 자료를 한 폴더로).

원칙:
- 카테고리는 가능한 한 **구체적인 프로젝트명/사업명/고객사명**으로 만든다.
  좋은 예: "한국지역정보개발원 초거대 AI 공통기반", "AVOCA 특허 명세서", "사숲 챗봇 RAG"
  피할 예: "문서", "보고서", "프레젠테이션", "업무 자료", "기타", "회사 문서"
- 파일명·본문에서 반복되는 **고유명사·약어·버전 패턴**(예: v1.0, R1, 240301)을 단서로 그룹을 묶는다.
  같은 사업의 v0.5, R1, 최종_4 같은 버전 파일들은 한 폴더로 묶는다.
- **분리 우선**: 사업명/프로젝트명이 분명히 다르면 묶지 말고 **별도 폴더**로 만든다.
  여러 사업을 무리하게 한 폴더로 묶지 마라. **유사도가 매우 높은 경우(같은 고객사의 같은
  사업, 동일 사업의 여러 버전·산출물)에만 묶는다.**
- 폴더 수가 많아져도 괜찮다 — 폴더명이 명확하고 그룹 번호가 잘 부여되어 있으면 사용자가
  쉽게 구분할 수 있다. 폴더 수 상한이 허용하는 한 **세분화**를 선호하라.
- **잡파일은 단 하나의 "9. 기타" 카테고리에만 모은다 — 절대 두 개 이상 만들지 마라.**
  사용 조건은 *정말로 분류하기 어려운 파일* 뿐. 두 파일 이상이 같은 사업/프로젝트로 묶일
  단서가 조금이라도 있으면 그 카테고리로 보내고 "기타"에 넣지 마라. 단발성 잡파일이라도
  의미가 식별되면 별도 카테고리로 만든다.
- 식별 가능한 프로젝트가 보이면 **프로젝트 폴더를 항상 더 우선**으로 만든다.
- **"사업·과제·프로젝트로 보이지만 단서가 약한" 파일은 "기타"가 아니라
  modified 시각이 가장 가까운 사업 폴더에 보내라.** 프로젝트 문서는 보통 그 사업 진행 기간에
  집중해서 만들어진다. 본문이나 파일명에 사업명이 명시되지 않았어도 modified 시각이
  특정 사업 카테고리의 시기 범위(예: "2024-Q1", "2023–2025") 안에 들면 그 카테고리에
  분류한다. reason에 "시기로 추정" 이라 명시하라.
  같은 시기에 사업 폴더가 여러 개라면 그 사업의 키워드 중 가장 자주 나오는 쪽으로.
  진짜 관련 사업이 하나도 없을 때만 "기타"에 넣는다.
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
- categories[].name 은 **구체적인 프로젝트/사업/고객사 이름** (시기 표기는 빼고 본질만).
- categories[].time_label 은 그 폴더에 모이는 파일들의 **시기 표기**를 한국어로 짧게.
  파일들의 modified 시각 분포를 직접 보고 가장 적합한 형태를 골라라:
    · 같은 달에 집중 → "2024-03"
    · 같은 분기 → "2024-Q1"  (예: 단기 제안·발표 작업)
    · 같은 반기 → "2024-H1"  (예: 한 학기 강의자료)
    · 1년 내 분산 → "2024"   (예: 1년짜리 사업/과제)
    · 여러 해에 걸친 장기 프로젝트 → "2023–2025" 또는 "2023~2025"
      (예: 다년 정부 사업, 박사논문 같이 수년간 누적되는 자료)
  모호하거나 한 파일뿐이면 빈 문자열 "".
- categories[].duration 은 폴더의 **기간 유형**을 다음 중 하나로 표기:
  "burst" (한 달 이내 집중 작업, 제안서/발표/단기 산출물),
  "short" (분기~반기 내, 단기 사업/이벤트),
  "annual" (한 해 동안의 사업/과제/연간 보고),
  "multi-year" (여러 해에 걸친 장기 프로젝트/연구),
  "mixed" (의미 있는 시기 패턴이 안 보이는 단발성 모음).
  duration은 file modified 시각의 **실제 분포**에 근거해 정한다 — 임의로 추측하지 마라.
- categories[].group 은 **관련성/주제별 묶음 번호**(1~9 정수, 0/누락 금지).
  같은 group 번호는 서로 비슷한 성격의 폴더끼리 부여한다. 예: 동일 고객사의 여러 사업이면
  같은 group, 내부 행사·잡파일류는 또 다른 group. 결과적으로 폴더 정렬을 도와준다.
  **모든 카테고리에 반드시 1~9 사이 숫자를 지정하라.** 잡파일/기타 카테고리는 9를 사용한다.
- assignments 의 path 는 입력으로 준 path 값을 **그대로** 복사한다(요약·축약·번역 금지).
- primary 는 categories[].id 중 하나여야 한다.
- primary_score 와 차이가 {ambiguity_threshold} 이하인 다른 후보가 있으면 secondary 에 넣는다.
- reason 은 어떤 프로젝트/단서로 묶었는지 한 줄(40자 이내).

응답 JSON 스키마(엄격, 추가 텍스트 금지):
{{
  "categories": [
    {{
      "id": "kebab-slug",
      "name": "구체 프로젝트/사업명",
      "description": "한 줄 설명",
      "time_label": "2024-Q1",
      "duration": "short",
      "group": 1
    }}
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


FILENAME_FIRST_PASS_INSTR = """다음은 코퍼스의 **파일명 목록**이다. 본문은 주지 않는다.

너는 두 가지 일을 동시에 한다:
1) 파일명만 보고 (a) 사업/과제/프로젝트/기관 축 + (b) 시스템 프롬프트에
   정의된 6개 *기능 축* (작업물·산출물 / 거래·계약 / 회계·금융 /
   회의·일정 / 원자료·미디어 / 외부 참고) 양쪽을 함께 고려해 카테고리를
   설계한다.  사업 식별이 불가능하면 기능 축 단독 폴더도 OK.
2) 각 파일에 대해 다음 중 하나를 선택한다:
   (a) **파일명만으로 어느 카테고리인지 확신**할 수 있다 → assignments 에 넣는다.
       파일명에 사업·고객사·시스템·제품·약어 등 식별자, 또는 기능을
       특정하는 단서(예: 견적/세금/회의 등 *기능적* 단어 — 이 예시들은
       너가 본 도메인에 따라 달라질 수 있음. 너 스스로 판단하라)가 분명히
       보이는 경우.
   (b) **파일명이 너무 일반적이거나 식별 단서가 없다** → assignments 에 넣지
       말고 **deferred** 배열에 그 파일의 path 그대로 넣는다.  본문을 보고
       다음 단계에서 다시 분류된다.

**즉시 deferred 로 보내야 할 파일명 형태 (분류 시도 자체 금지):**
  · 의미 토큰이 0 인 파일 — 순수 숫자, 랜덤 해시/인코딩 문자열,
    카메라·녹화기 자동 명명(IMG/DSC/Screenshot/Recording 류),
    미디어 확장자이면서 파일명에 의미 토큰이 없는 경우 등.
이런 파일은 *어떤 추측으로도* 사업·기능 폴더에 confident 하게 넣을 수 없다.

**키워드 일치 조건 (assignments 의 hard rule):**
  파일명 토큰 중 *최소 한 개*가 그 카테고리의 name·keywords·description
  토큰과 의미적으로 겹쳐야만 assignments 에 넣는다.  겹치는 토큰이 없으면
  주제가 비슷해 보여도 deferred 로 보낸다.

원칙:
- 추상 라벨 금지 ("문서", "보고서", "프레젠테이션", "기타").
- 확신이 70% 이하면 deferred 로 보낸다 — 잘못된 분류보다 다음 단계로 넘기는 것이 낫다.
- 같은 사업의 파일이라도 *기능*이 다르면 (작업물 vs 거래 vs 회계 vs 회의 vs
  미디어 vs 외부 참조) 별도 카테고리로 분리한다.  사업명 일치만으로 묶지
  마라 — 한 폴더는 한 사업 × 한 기능의 교집합이다.
- categories 수는 {min_categories}~{max_categories} 사이로 둔다.

응답 JSON 스키마(엄격, 다른 텍스트 금지):
{{
  "categories": [
    {{ "id": "slug","name": "구체 폴더명","description": "한 줄","time_label": "","duration": "mixed","group": 1 }}
  ],
  "assignments": [
    {{ "path": "원본 path","primary": "category_id",
       "primary_score": 0.0,
       "secondary": [ {{"id": "category_id","score": 0.0}} ],
       "reason": "한 줄 사유" }}
  ],
  "deferred": [ "원본 path", "원본 path" ]
}}
임계값({ambiguity_threshold}) 이하로 차이나는 다른 카테고리만 secondary 에 추가."""


def build_filename_first_pass(
    files: list[dict],
    min_categories: int,
    max_categories: int,
    ambiguity_threshold: float,
    *,
    reclassify_mode: bool = True,
    classification_guidance: str = "",
) -> str:
    # files: [{"path": "...", "name": "...", "ext": "..."}].  No body.
    body = json.dumps({"files": files}, ensure_ascii=False, indent=2)
    instr = FILENAME_FIRST_PASS_INSTR.format(
        min_categories=min_categories,
        max_categories=max_categories,
        ambiguity_threshold=ambiguity_threshold,
    )
    return f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\n데이터:\n{body}"


LONGTAIL_DISCOVER_INSTR = """이 파일들은 1차 클러스터링에서 어느 사업/과제/프로젝트/기관/목적/용도에도
정합하지 못해 **롱테일**로 분리된 파일들이다. 위 'categories'는 1차에서 이미 합의된 폴더 목록이고,
아래 'files' 는 거기에 어울리지 않거나 본문 유사도가 낮아 빠진 파일들이다.

각 파일에 대해 둘 중 하나만 선택하라:
  (a) 'categories' 중 정말 어울리는 것이 있다 → primary 에 그 id 를 넣는다.
  (b) 어울리는 것이 없다 → 새 카테고리를 *너가 직접 제안*하고 거기에 넣는다.
      새 카테고리는 응답의 'new_categories' 배열에 추가하고, primary 에 그 id 를 쓴다.
      새 카테고리 또한 사업/과제/프로젝트/기관/목적/용도 단위로 만들고, 추상적 라벨
      ("문서", "보고서", "기타")은 금지. 같은 새 id 를 여러 파일이 공유해도 좋다.

응답 JSON 스키마(엄격):
{{
  "new_categories": [
    {{ "id": "slug","name": "구체 폴더명","description": "한 줄","time_label": "","duration": "mixed","group": 1 }}
  ],
  "assignments": [
    {{ "path": "원본 path", "primary": "category_id",
       "primary_score": 0.0,
       "secondary": [ {{ "id": "category_id", "score": 0.0 }} ],
       "reason": "한 줄 사유" }}
  ]
}}
임계값({ambiguity_threshold}) 이하로 차이나는 다른 카테고리만 secondary 에 추가."""


def build_longtail_discover(
    files: list[dict],
    categories: list[dict],
    ambiguity_threshold: float,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    body = json.dumps(
        {"categories": categories, "files": files}, ensure_ascii=False, indent=2
    )
    instr = LONGTAIL_DISCOVER_INSTR.format(ambiguity_threshold=ambiguity_threshold)
    return f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\n데이터:\n{body}"


RECLASSIFY_HINT = (
    "사용자가 명시적으로 *재분류*를 요청했다. "
    "각 파일 path 의 부모 폴더 컴포넌트는 의도적으로 `[folder]` 자리표시자로 가려 두었다. "
    "기존 폴더 그룹은 신뢰할 수 없으니, 파일명과 본문 발췌만 보고 새 카테고리를 직접 설계하라. "
    "비슷한 본문 내용이 다른 폴더에 흩어져 있을 수 있으니 적극적으로 그룹을 재구성해도 좋다."
)


def _maybe_hint(reclassify_mode: bool) -> str:
    return f"\n\n{RECLASSIFY_HINT}" if reclassify_mode else ""


def build_single_call(
    files: list[dict],
    min_categories: int,
    max_categories: int,
    ambiguity_threshold: float,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    instr = SINGLE_CALL_INSTRUCTION.format(
        min_categories=min_categories,
        max_categories=max_categories,
        ambiguity_threshold=ambiguity_threshold,
    )
    body = json.dumps({"files": files}, ensure_ascii=False, indent=2)
    return f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\n데이터:\n{body}"


def build_stage_a(
    files: list[dict], *, reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    body = json.dumps(files, ensure_ascii=False, indent=2)
    return (
        f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n"
        f"{STAGE_A_INSTRUCTION}\n\n파일 목록:\n{body}"
    )


def build_stage_merge(
    candidate_sets: list[list[dict]],
    min_categories: int,
    max_categories: int,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    merged = {"batches": [{"candidates": cs} for cs in candidate_sets]}
    body = json.dumps(merged, ensure_ascii=False, indent=2)
    instr = STAGE_MERGE_INSTRUCTION.format(
        min_categories=min_categories, max_categories=max_categories
    )
    return f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\n후보 묶음:\n{body}"


def build_stage_b(
    files: list[dict],
    categories: list[dict],
    ambiguity_threshold: float,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    instr = STAGE_B_INSTRUCTION.format(ambiguity_threshold=ambiguity_threshold)
    body = json.dumps(
        {"categories": categories, "files": files}, ensure_ascii=False, indent=2
    )
    return f"{STAGE_A_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\n데이터:\n{body}"


# -----------------------------------------------------------------------
# Compact prompts used by the "local LLM" micro-batch path so total token
# count per call stays small even with 4k–8k context windows.
# -----------------------------------------------------------------------


COMPACT_SYSTEM = """너는 파일을 프로젝트/사업 단위로 묶는 전문가다.
파일명에 반복적으로 나타나는 고객사명, 사업명, 시스템명, 약어, 버전 패턴을 찾아라.
폴더명은 구체적으로 작성한다. "문서", "보고서", "프레젠테이션" 같은 추상 라벨은 금지.
"""

COMPACT_DISCOVER_INSTR = """이 파일들에서 잘게 쪼갠 카테고리 후보만 뽑아라.
응답은 정확히 다음 JSON, 다른 텍스트 금지:
{
  "candidates": [
    { "id": "kebab-slug", "name": "구체 이름", "keywords": ["힌트", "단서"] }
  ]
}"""


COMPACT_MERGE_INSTR = """여러 배치에서 모은 카테고리 후보를 통합하라.
- 의미가 비슷하면 합친다.
- 최종 {min_categories}~{max_categories}개로 줄인다.
- categories[].group(1~9), time_label, duration 도 부여하라.
- duration ∈ {{burst, short, annual, multi-year, mixed}}, time_label은
  duration에 맞춰 "2024-03" / "2024-Q1" / "2024-H1" / "2024" / "2023–2025" 형식.
응답 JSON:
{{
  "categories": [
    {{ "id":"slug","name":"구체 이름","description":"한 줄","time_label":"","duration":"mixed","group":1 }}
  ]
}}"""


COMPACT_ASSIGN_INSTR = """주어진 categories 만 사용하여 각 파일을 분류하라.
응답 JSON:
{{
  "assignments": [
    {{ "path":"입력 path 그대로", "primary":"category_id",
       "primary_score":0.0,
       "secondary":[ {{"id":"category_id","score":0.0}} ],
       "reason":"한 줄 사유" }}
  ]
}}
임계값({ambiguity_threshold}) 이하로 차이나는 다른 카테고리만 secondary 에 추가."""


def build_compact_discover(
    files: list[dict], *, reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    body = json.dumps(files, ensure_ascii=False)
    return (
        f"{COMPACT_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n"
        f"{COMPACT_DISCOVER_INSTR}\n\nfiles:{body}"
    )


def build_compact_merge(
    candidate_sets: list[list[dict]],
    min_categories: int,
    max_categories: int,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    body = json.dumps({"batches": candidate_sets}, ensure_ascii=False)
    instr = COMPACT_MERGE_INSTR.format(
        min_categories=min_categories, max_categories=max_categories
    )
    return f"{COMPACT_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\nbatches:{body}"


def build_compact_assign(
    files: list[dict],
    categories: list[dict],
    ambiguity_threshold: float,
    *,
    reclassify_mode: bool = False,
    classification_guidance: str = "",
) -> str:
    body = json.dumps({"categories": categories, "files": files}, ensure_ascii=False)
    instr = COMPACT_ASSIGN_INSTR.format(ambiguity_threshold=ambiguity_threshold)
    return f"{COMPACT_SYSTEM}{_maybe_hint(reclassify_mode)}{_guidance_block(classification_guidance)}\n\n{instr}\n\ndata:{body}"
