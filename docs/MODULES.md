# FolderAngel — Module Specification

각 모듈은 **단일 책임**을 갖고, **아래 계약(Contract)을 절대 위반하지 않는다**. 계약 위반은 버그로 간주한다.

## M1. `folderangel.config` — 설정·경로 관리
- **책임**: 앱 전용 디렉토리(`~/.folderangel`)를 보장. 설정 로드/세이브. API 키 keyring 연동.
- **공개 API**:
  - `AppPaths` dataclass(`root`, `config`, `index_db`, `logs_dir`)
  - `load_config() -> Config`, `save_config(cfg: Config)`
  - `get_api_key() -> str | None`, `set_api_key(key: str)`
- **불변식**: 모든 경로는 `pathlib.Path`, 존재하지 않으면 자동 생성.
- **에러**: keyring 실패 시 경고 반환하지만 예외는 삼킨다.

## M2. `folderangel.scanner` — 파일 스캐너
- **입력**: `root: Path`, `recursive: bool`, `ignore_patterns: list[str]`
- **출력**: `Iterable[ScannedFile]` (lazy)
- **규칙**: symlink 비주행. `ignore_patterns`는 `fnmatch` 패턴. 최대 파일 수 `max_files` 초과 시 `ScanTooLargeError`.

## M3. `folderangel.metadata` — 메타데이터 추출
- **입력**: `Path`
- **출력**: `FileEntry`(name, ext, size, created, modified, accessed, mime)
- **크로스플랫폼**: Windows에서 `ctime`을 작성일로 간주, POSIX에서는 `st_birthtime` 또는 `st_ctime`.

## M4. `folderangel.parsers` — 문서 본문 추출기
- **디스패처**: `extract_excerpt(path, max_chars=1800, timeout=5.0) -> str`
- **확장자별 구현**:
  - `pdf` → `pypdf` 페이지 순회, `max_chars` 도달 시 중단.
  - `docx` → `python-docx` paragraphs 순회.
  - `pptx` → `python-pptx` 슬라이드 텍스트 순회.
  - `xlsx` → `openpyxl` 시트명 + 처음 행들(옵션).
  - `hwpx` → zipfile + `Contents/section*.xml` 텍스트.
  - `hwp` → `olefile`로 BodyText 스트림 읽고 텍스트 후보 추출(heuristic).
  - `rtf` → 정규식 기반 텍스트 스트립.
  - `txt`/`md`/`csv`/`odt` → 직접 읽기(odt는 zip+content.xml).
- **실패 시**: 빈 문자열 반환 + 로거 경고.
- **순수성**: 파일 수정·이동하지 않는다.

## M5. `folderangel.llm` — LLM 플래너
- **M5.1 `client.GeminiClient`**
  - `generate_json(prompt: str, schema: dict) -> dict`
  - 모델 기본값 `gemini-2.5-flash`
  - REST `v1beta/models/{model}:generateContent`
  - 재시도 1회, 60초 타임아웃
  - **스키마 검증 실패 시 예외 throw** — 상위에서 Mock으로 폴백
- **M5.2 `mock.MockPlanner`** — 키 없거나 LLM 실패 시 휴리스틱(확장자 그룹핑 + 파일명 키워드)
- **M5.3 `prompts`** — 프롬프트 템플릿 상수
- **입출력 계약**
  - Stage A: 파일 배치 → `{candidates: [{id, name, description, keywords}]}`
  - Stage A-merge: 모든 candidates → `{categories: [{id, name, description}]}` (3–12개)
  - Stage B: 파일 배치 + categories → `{assignments: [{path, primary, primary_score, secondary:[{id,score}], reason}]}`

## M6. `folderangel.planner` — 전체 플래닝 오케스트레이터
- **책임**: 파일 목록과 카테고리 범위를 받아 `Plan`(카테고리 + 파일별 assignment 리스트)을 생성.
- **배치**: 기본 30 파일/배치. 타이밍/토큰 제한 반영.
- **폴백 규칙**: 어떤 단계든 실패 시 Mock으로 해당 배치 보강.

## M7. `folderangel.organizer` — 실행기
- **입력**: `Plan`, `dry_run: bool`
- **동작**:
  1. 카테고리별 폴더 생성(이름 정규화).
  2. 각 Assignment에 대해 파일 이동(`shutil.move`), 충돌 시 자동 리넘버.
  3. Secondary 카테고리가 있으면 해당 폴더에 바로가기 생성(shortcuts 모듈 사용).
  4. 결과 `OperationResult` 반환.
- **안전장치**: `dry_run=True`이면 계획만 산출하고 디스크 변경 없음.

## M8. `folderangel.shortcuts` — OS 바로가기
- **Linux**: symlink(`os.symlink(target, link_path)`)
- **Windows**: `.lnk` 생성. `pywin32`가 있으면 IShellLink 사용, 없으면 PowerShell 스크립트로 `WScript.Shell`의 `CreateShortcut` 호출, 둘 다 불가하면 `.url` fallback.
- **API**: `create_shortcut(target: Path, link_path: Path) -> Path`

## M9. `folderangel.index` — SQLite 인덱스 & 검색
- **스키마**: SPEC §5.4 준수.
- **API**:
  - `record_operation(op: OperationResult) -> op_id`
  - `search(query: str, limit=50) -> list[SearchHit]`
  - `list_operations(limit=50) -> list[OperationInfo]`
  - `rollback(op_id) -> RollbackResult`
- **FTS**: FTS5 virtual table + 트리거로 자동 동기화.

## M10. `folderangel.reporter` — 리포트
- **API**: `emit_markdown(op: OperationResult, out_dir: Path) -> Path`
- **내용**: 요약, 카테고리 분포, 이동 목록, 스킵 목록, 바로가기 목록, 트리 다이어그램(최대 500 파일).

## M11. `folderangel.worker` — 백그라운드 실행자
- **책임**: UI 스레드와 분리된 `QThread` 내에서 파이프라인 실행.
- **시그널**: `scanned`, `parsed(file)`, `plan_ready`, `organized(file)`, `finished(op_result)`, `error(msg)`, `canceled`.
- **취소 프로토콜**: `cancel()` 호출 시 다음 안전 지점에서 정상 종료.

## M12. `folderangel.ui` — PySide6 UI
- **엔트리**: `ui.main:launch()`
- **QSS**: `ui/styles.py`에 라이트/다크 모드 QSS 템플릿.
- **뷰**: OrganizeView, SearchView, HistoryView, SettingsView.
- **공통 위젯**: PathBar, ProgressSteps, ReportCard, SearchList.
- **시그널 연결**: Worker → Views via Qt signals.

## M13. `folderangel.__main__` — CLI 진입점
- `python -m folderangel` → UI 실행.
- `python -m folderangel --cli --path ... [--recursive] [--dry-run]` → 헤드리스 실행(검증/CI용).

---

## 모듈 테스트 매트릭스

| 모듈 | 단위 테스트 | 통합 테스트 |
| --- | --- | --- |
| config | 경로 생성, 기본 구성 | - |
| scanner | 재귀/비재귀, 무시 패턴, symlink 스킵 | full pipeline |
| metadata | 확장자·size·mime 검증 | full pipeline |
| parsers | 샘플 문서 6개 고정 fixture | full pipeline |
| llm.mock | 휴리스틱 결정성 | full pipeline |
| planner | batch 분할, 폴백 경로 | full pipeline |
| organizer | 충돌 리넘버, dry-run, shortcut | full pipeline |
| shortcuts | Linux symlink 생성, Windows 로직은 모킹 | - |
| index | insert + fts 쿼리 + rollback | - |
| reporter | markdown 생성, 트리 렌더 | - |
| worker | cancel 시점 처리 | - |
| ui | QSS 로드 smoke | - |

## 2회 버그 수정 루프 규칙
각 모듈은 구현 직후 `pytest tests/test_<mod>.py` 자동 실행. 실패 시:
1. 에러 메시지에 기반해 **자가 패치**(최대 200줄 범위).
2. 재실행 실패하면 **대체 구현**(단순화된 폴백 경로).
3. 두 번 모두 실패하면 해당 기능에 **graceful-degradation 표시**를 남기고 리포트에 경고를 출력.
