# FolderAngel — Functional & UI Specification

| 항목 | 내용 |
| --- | --- |
| 문서 버전 | 1.0 |
| 최종 수정 | 2026-04-25 |
| 작성자 | FolderAngel Core Team |
| 제품명 | FolderAngel |
| 플랫폼 | Linux (X11/Wayland), Windows 10+ |
| 런타임 | Python 3.11+ / PySide6 |
| 배포 형태 | 소스, PyInstaller 단일 실행 파일 |

---

## 1. 제품 개요

### 1.1 목적
FolderAngel은 사용자가 지정한 폴더 내부의 파일들을 **LLM 기반으로 의미 분류**하여 **직관적인 하위 폴더로 자동 정리**해 주는 크로스플랫폼 데스크톱 애플리케이션이다. 단순 확장자/규칙 기반이 아닌 **파일명·메타데이터·문서 본문 요약**을 종합해 분류하므로, 사람이 수동으로 정리한 것과 유사한 수준의 폴더 체계를 생성한다.

### 1.2 배경
개인/업무용 다운로드 폴더, 자료 아카이브, 강의 자료 등은 시간이 지나면 수백 수천 개 파일로 누적된다. 기존 규칙 기반 툴(File Juggler, Maid 등)은 확장자·이름 패턴 수준에 그쳐 맥락을 반영하지 못한다. FolderAngel은 LLM을 **폴더 구조 결정**과 **분류 계획 수립** 단계에만 제한적으로 사용하여, 비용 통제와 설명가능성을 유지하면서 의미 기반 정리를 제공한다.

### 1.3 비목표 (Non-Goals)
- 파일 내용 자체의 편집·요약·번역 기능.
- 클라우드 드라이브 동기화 기능.
- 대용량(GB 단위) 단일 파일 처리 최적화.
- LLM을 이용한 파일 이름 변경 (이번 버전에서는 폴더명과 분류에만 사용).

---

## 2. 이해관계자 & 유스케이스

### 2.1 주요 사용자
- **P1 지식노동자**: 다운로드/문서 폴더 정리. 월 1~2회 일괄 정리.
- **P2 연구자·학생**: 논문·강의자료 아카이브 정리. 본문 파싱 정확도 중요.
- **P3 크리에이터/디자이너**: 수많은 에셋 파일 정리. 파일명·메타 기반 분류 필요.

### 2.2 핵심 유스케이스
| UC | 이름 | 흐름 |
| --- | --- | --- |
| UC-01 | 폴더 스캔 & 정리 | 폴더 선택 → 하위 포함 여부 선택 → 스캔 → LLM 플랜 확인 → 실행 → 리포트 |
| UC-02 | 중복 분류 처리 | 유사 점수 두 폴더 이상 → 최고 점수 폴더로 이동, 나머지는 바로가기 |
| UC-03 | 사후 검색 | 원본 파일명·폴더명으로 FTS 검색 → 현재 위치 표시 |
| UC-04 | 미리보기/건식 실행 (Dry-Run) | 계획만 수립, 실제 이동 없이 리포트만 생성 |
| UC-05 | 롤백 | 최근 정리 작업을 원복(원본 경로로 복구) |
| UC-06 | API 키 설정 | 설정에서 Gemini API 키 입력·검증·저장 |

---

## 3. 기능 요구사항 (Functional Requirements)

표기 규칙: **FR-[영역]-[번호]** / **우선순위 M=Must, S=Should, C=Could**.

### 3.1 대상 선택 & 스캔
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-SCAN-01 | 사용자는 단일 폴더를 선택할 수 있다. | M |
| FR-SCAN-02 | 하위 폴더 포함 여부(재귀)를 토글할 수 있다. | M |
| FR-SCAN-03 | 숨김/시스템/임시 파일(`.*`, `~$*`, `Thumbs.db`, `.DS_Store`)은 기본 제외. 설정에서 해제 가능. | M |
| FR-SCAN-04 | 심볼릭 링크는 따라가지 않는다(무한 루프 방지). | M |
| FR-SCAN-05 | 최대 파일 수 한도(기본 5,000)를 초과하면 경고 후 진행/취소 선택. | S |
| FR-SCAN-06 | 파일당 기본 메타데이터(이름, 확장자, 크기, 작성일, 수정일, 접근일, MIME 추정)를 수집한다. | M |

### 3.2 문서 본문 파싱
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-PARSE-01 | 지원 형식: `pdf`, `docx`, `pptx`, `xlsx`, `hwp`, `hwpx`, `txt`, `md`, `rtf`, `csv`, `odt`. | M |
| FR-PARSE-02 | 문서 앞부분 **A4 약 1장 분량**에 해당하는 본문을 추출한다(기준: 최대 1,800자, 파싱 실패 시 공백). | M |
| FR-PARSE-03 | 파싱 실패 시 메타데이터만으로 처리하며 에러 로그에 기록한다. | M |
| FR-PARSE-04 | 파일당 파싱 타임아웃 5초. 초과 시 스킵. | S |
| FR-PARSE-05 | 이미지(OCR)·오디오·비디오 본문 추출은 대상이 아니다. 메타만 사용. | C |

### 3.3 LLM 호출 & 분류 계획
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-LLM-01 | 기본 LLM은 Google Gemini 2.5 Flash. 사용자가 API 키를 입력하지 않으면 **Mock Planner**(휴리스틱)로 폴백. | M |
| FR-LLM-02 | 파일 수가 많으면 **배치 분할**한다(기본 30 파일/배치). | M |
| FR-LLM-03 | 플래닝은 2단계이다. **Stage A**: 각 배치에서 카테고리 후보 수집 → **Stage A-merge**: 후보 통합 → **Stage B**: 최종 카테고리 기준으로 각 파일 분류. | M |
| FR-LLM-04 | 최종 카테고리 수는 기본 3~12개 범위로 제한하고 LLM이 최적 개수를 결정한다. | M |
| FR-LLM-05 | LLM 응답은 반드시 JSON 스키마를 준수해야 하며, 파싱 실패 시 1회 재시도 후 Mock으로 폴백. | M |
| FR-LLM-06 | 파일당 토큰 소비를 제한하기 위해 본문 요약은 1,800자(≈600 토큰 KR 기준)로 절단 후 전송. | M |
| FR-LLM-07 | 사용자는 LLM 호출 총 토큰/비용 추정을 실행 전에 확인할 수 있다. | S |

### 3.4 폴더 생성 & 이동
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-ORG-01 | 최종 카테고리마다 대상 폴더 하위에 폴더를 생성한다. | M |
| FR-ORG-02 | 폴더명은 한국어/영어 혼용 허용, OS 금지 문자(`\/:*?"<>|`)는 자동 치환. | M |
| FR-ORG-03 | 이름 충돌 시 `기존파일명 (2).ext` 규칙으로 자동 넘버링. | M |
| FR-ORG-04 | 이동은 **원자적**으로 처리하되 실패 시 해당 파일은 원위치 유지하고 리포트에 기록. | M |
| FR-ORG-05 | **모호 분류(멀티 소속)**: top1 점수와 top2 점수 차가 설정 임계값(기본 0.15) 이하일 경우, top1에 실 이동하고 나머지 후보 폴더에는 OS별 **바로가기(Linux: symlink, Windows: .lnk)** 를 생성한다. | M |
| FR-ORG-06 | 바로가기는 `원본파일명.lnk` 또는 `원본파일명` 심볼릭 링크로 생성하며, 대상은 최종 이동된 파일의 **절대 경로**를 가리킨다. | M |
| FR-ORG-07 | 모든 원본 경로·대상 경로를 인덱스에 기록하여 롤백·검색에 사용한다. | M |

### 3.5 인덱스 & 검색
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-IDX-01 | 실행 결과는 앱 전용 SQLite DB(`~/.folderangel/index.db`)에 저장한다. | M |
| FR-IDX-02 | SQLite FTS5를 사용하여 파일명·폴더명·카테고리·원본 경로를 전문 검색 가능. | M |
| FR-IDX-03 | 검색 결과는 원본 경로, 현재 위치, 정리 시각, 카테고리, 사유(LLM reason)를 포함한다. | M |
| FR-IDX-04 | 롤백 기능은 인덱스에 기록된 `operation_id` 단위로 동작한다. | S |

### 3.6 리포트
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-REP-01 | 실행 후 **요약 리포트**(총 파일 수, 이동 수, 바로가기 수, 스킵 수, 카테고리별 파일 분포)를 화면과 Markdown 파일(`{target}/FolderAngel_Report_{timestamp}.md`)로 생성한다. | M |
| FR-REP-02 | 리포트는 **전/후 파일 트리**를 포함한다(최대 500 파일 범위). | S |
| FR-REP-03 | 각 파일에 대해 **분류 사유**(LLM reason)를 포함한다. | S |

### 3.7 설정
| ID | 요구사항 | 우선 |
| --- | --- | --- |
| FR-CFG-01 | 설정은 `~/.folderangel/config.json`에 보관한다. | M |
| FR-CFG-02 | Gemini API 키는 OS keyring이 가능하면 keyring에, 실패 시 config.json 평문에 저장(경고 표시). | M |
| FR-CFG-03 | 사용자는 모델명, 배치 크기, 카테고리 범위, 모호 임계값, 본문 최대 길이를 조정할 수 있다. | S |

---

## 4. 비기능 요구사항 (Non-Functional)

| ID | 요구사항 |
| --- | --- |
| NFR-01 | **크로스플랫폼**: Linux(glibc 2.34+), Windows 10 1809+에서 동일 코드로 동작. |
| NFR-02 | **응답성**: 1,000 파일 스캔 30초 이내(본문 파싱 포함, SSD 기준). UI는 워커 스레드를 사용해 멈추지 않는다. |
| NFR-03 | **안정성**: 어느 파일 처리 실패도 전체 작업을 중단시키지 않는다(스킵 후 리포트 기록). |
| NFR-04 | **프라이버시**: LLM에 전송되는 텍스트는 "파일명, 확장자, 날짜, 1,800자 요약"으로 한정. 본문 원본을 서버에 저장하지 않는다. 옵트인 로그만 기록. |
| NFR-05 | **접근성**: 최소 1280×720 해상도, 다크/라이트 모드 자동 전환, 키보드 전용 조작 가능. |
| NFR-06 | **국제화**: 한국어/영어 UI. 기본 ko-KR. |
| NFR-07 | **로깅**: `~/.folderangel/logs/`에 rotating 로그. 레벨: INFO/ERROR. |

---

## 5. 데이터 모델

### 5.1 파일 엔트리 (`FileEntry`)
```
path: str            # 원본 절대경로
name: str
ext: str             # .pdf 포함
size: int
created: ISO-8601
modified: ISO-8601
accessed: ISO-8601
mime: str
content_excerpt: str # 최대 1800자
```

### 5.2 카테고리 (`Category`)
```
id: str              # 슬러그 (ko-ascii)
name: str            # 사람이 읽기 쉬운 폴더명
description: str     # LLM이 붙인 한 줄 설명
```

### 5.3 분류 결과 (`Assignment`)
```
file_path: str
primary_category_id: str
primary_score: float    # 0..1
secondary: [(category_id, score), ...]  # 임계값 이하 차이 후보들
reason: str
```

### 5.4 인덱스 DB
```
operations(id, target_root, started_at, finished_at, dry_run, stats_json)
files(id, op_id, original_path, new_path, category, reason, score, created_at)
shortcuts(id, op_id, file_id, shortcut_path)
fts(files_fts: filename, folder, category, reason, original_path)
```

---

## 6. 처리 파이프라인

```
① Select folder (UI)
② Scan + recursion flag → FileEntry[]
③ For each file (parallel, thread pool):
      metadata + parse excerpt
④ PlannerStageA(batches) → category_candidates[]
⑤ PlannerStageAMerge(candidates) → final_categories[]
⑥ PlannerStageB(file_batches, final_categories) → Assignment[]
⑦ Organizer.execute(Assignments, dry_run?)
⑧ IndexWriter.record(op, files, shortcuts)
⑨ Reporter.emit(op)
```

---

## 7. 오류 처리 전략

| 시나리오 | 대응 |
| --- | --- |
| LLM 키 없음 | Mock 플래너로 폴백, UI에 배너 표시 |
| LLM 호출 타임아웃(60초) | 1회 재시도 → Mock 폴백 |
| JSON 스키마 불일치 | 정규화 시도 → 실패 시 해당 배치 버리고 Mock 보강 |
| 파일 이동 권한 오류 | 스킵 + 리포트 |
| 디스크 공간 부족 | 실행 전 체크(예상 이동 용량 vs 가용 용량) |
| 중복 실행 방지 | 동일 폴더 대상 작업은 파일 락으로 1개만 허용 |

---

## 8. UI 명세

### 8.1 디자인 원칙
- **애플 감성**: 깔끔한 여백, 24~32pt 타이틀, SF Pro/Apple SD/Pretendard 느낌의 시스템 폰트 우선 사용, 라운드 코너(12px), 은은한 그림자, 시스템 컬러(accent blue #007AFF).
- **3클릭 룰**: 폴더 선택 → 옵션 확인 → 실행 버튼. 최대 3 클릭으로 작업 시작.
- **Live Preview**: 계획 단계에서 카테고리별 파일 수가 실시간 표시.
- **되돌리기 우선**: 실행 후 상단에 "되돌리기" 버튼이 한눈에 보이도록 배치.

### 8.2 화면 구성
- **메인 창 (1280×800 기본)**: 좌측 사이드바 + 우측 워크스페이스 레이아웃.
  - 사이드바 탭: ① Organize(홈), ② Search, ③ History, ④ Settings.
- **Organize 탭**:
  - 상단 Path Bar: 폴더 선택 버튼 + 경로 표시 + "하위 폴더 포함" 토글.
  - 중단 Options 카드: 모드(Dry-Run / Execute), 카테고리 수 범위 슬라이더.
  - Primary CTA: "정리 시작" (accent blue 필, 44pt 높이).
  - 실행 중: 단계별 진행 인디케이터(Scan → Parse → Plan → Organize) + 프로그레스 바 + 현재 처리 파일.
  - 완료: 리포트 카드(파일 수, 카테고리 수, 바로가기 수 + "리포트 열기" + "되돌리기").
- **Search 탭**: 검색창(포커스 단축키 Ctrl/⌘+F) + 결과 리스트(파일명, 카테고리, 현재 경로, 원본 경로, 일시).
- **History 탭**: 최근 정리 작업 리스트(시간, 대상 폴더, 파일 수). 행 클릭 시 리포트 보기/롤백.
- **Settings 탭**: API 키 입력(비밀번호 입력 스타일), 모델, 배치 크기, 모호 임계값, 언어, 데이터 경로.

### 8.3 상호작용 규칙
- 모든 장시간 작업은 취소 가능해야 한다(Worker thread + cancel flag).
- Esc → 현재 패널 취소/닫기.
- Enter → primary CTA.
- Ctrl/⌘+, → Settings.
- 드래그 앤 드롭: 폴더를 Organize 탭에 드롭하면 경로 자동 선택.

### 8.4 상태 표시·색상
| 상태 | 색상 |
| --- | --- |
| 성공 | #34C759 |
| 주의 | #FF9500 |
| 에러 | #FF3B30 |
| 정보/링크 | #007AFF |
| 배경(라이트) | #F5F5F7 |
| 배경(다크) | #1C1C1E |

---

## 9. 릴리즈 범위 (v1.0 MVP)

- **포함**: FR-SCAN-01~06, FR-PARSE-01~04, FR-LLM-01~06, FR-ORG-01~07, FR-IDX-01~03, FR-REP-01~03, FR-CFG-01~02, 전체 NFR-01~07, UI Organize/Search/History/Settings.
- **이후**: 롤백(FR-IDX-04)은 데이터는 기록하되 UI 완성도는 베타 수준; 토큰 비용 프리뷰(FR-LLM-07), 세부 설정(FR-CFG-03)은 v1.1.

---

## 10. 테스트 전략 (요약)
- **Unit**: scanner/parser/index/organizer/shortcuts 각각의 pytest 케이스 + 크로스플랫폼 분기 모킹.
- **Integration**: 샘플 문서 40여 개로 full pipeline 실행(Mock LLM) → 리포트 검증.
- **Smoke**: 실제 Gemini API 키 1회 end-to-end(선택).
- **2회 버그 픽스 루프**: 각 모듈은 초기 구현 후 자동 테스트 실행 → 실패 시 스스로 수정 → 최대 2회 반복.
