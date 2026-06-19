# HADO 리듬 게임 (HADO Rhythm Game)

> Just Dance × HADO 리듬 — 화면에 표시된 동작을 80초 동안 얼마나 많이 따라하는지로 점수를 매기는 인터랙티브 동작 인식 게임.
>
> 박진우 (JINU) · 7년 하도 선수 경력 · 2024 HADO KOREA CUP 「하도리듬」 공식 시연 · HUFS IoT Spring 2026 최종발표용

---

## 🎯 무엇을 하는 시스템인가

- 카메라가 사용자의 17 keypoints를 실시간 추출 (YOLOv8n-pose)
- 8가지 **HADO 기본 동작**(하도리듬운동트레이닝 기반)을 분류
- 화면에 **"수행할 다음 동작"** 표시 → 사용자가 따라하면 점수 +1
- 80초 동안 몇 개를 수행했는지로 점수 부여
- 동작별 평균 소요 시간, 가장 많이 한 동작 등 결과 요약 출력

---

## 🤸 학습된 8가지 HADO 기본 동작

| 기호 | 동작 (영문 / 한글) | 설명 |
|------|-----------------|------|
| ⬇ | `squat` / 스쿼트 | 양발 넓게, 무릎 굽혀 낮춤 |
| ↔ | `side_step` / 사이드스텝 | 측면 소폭 이동, 직립 |
| ↗ | `lunge` / 런지 | 한발 앞, 앞무릎 90도 |
| ↔ | `slide` / 슬라이드 | 측면 넓게, 한쪽 무릎 깊게 굽힘 |
| ⚡ | `running_slide` / 런닝슬라이드 | 동적 측면이동 + 지면 터치 |
| ▼ | `burpee` / 버피테스트 | 바닥 플랭크 ↔ 기립 전환 |
| ★ | `rhythm_box` / 하도리듬박스 | 무릎 높이 킥 + 팔 동작 콤보 |
| ● | `ready` / 준비 자세 | 기본 직립 (게임 타겟에서는 제외) |

학습: 202 샘플 / 16 feature / sklearn classifier (`models/hado_movement_clf.pkl`)

---

## 🎮 게임 모드 (Just Dance × HADO 리듬)

```bash
./run.sh game                       # 기본: NORMAL, 80초
./run.sh game --level EASY          # 4동작만, 초보용
./run.sh game --level HARD          # 중급 동작 추가 후 사용 (현재 NORMAL과 동일)
./run.sh game --duration 60         # 60초 모드
./run.sh game_record                # 80초 자동 시작 + 영상 녹화 (발표용)
./run.sh game_live --threaded       # Pi 4 라이브 (NCNN + 스레드 캡처)
```

**조작:**
- `SPACE` — 시작
- `R` — 결과 화면에서 재시작
- `ESC` — 종료

**점수 시스템:**
- 타겟 동작이 표시됨 → 사용자 수행
- 6프레임(≈0.4초) 연속 매치 + 신뢰도 ≥ 60% → 성공, 점수 +1
- 0.4초 시각 피드백(초록 플래시 + `+1`) → 0.8초 쿨다운 → 새 타겟
- 80초 종료 시 총 점수 + 동작별 통계 + 평균 소요 시간 출력

---

## 🚀 빠른 시작

### 1. 환경 설치

```bash
python3 -m venv hado_venv
source hado_venv/bin/activate
pip install -r requirements.txt
```

### 2. 게임 실행

```bash
./run.sh game                        # 노트북 웹캠
./run.sh game --source 1             # 외장 카메라
./run.sh game --record demo.mp4      # 화면 녹화
```

### 3. (선택) 모델 재학습

8동작 모델은 이미 학습돼있음. 사진을 추가하거나 중급 동작을 학습하려면:

```bash
./run.sh label-movements       # 새 사진 라벨링
./run.sh train-model           # 재학습 → models/hado_movement_clf.pkl 갱신
```

---

## 🗂 폴더 구조

```
hado-rhythm-game/
├── src/
│   ├── camera.py                # 카메라 추상화 (picamera2 / OpenCV, 스레드)
│   ├── detector.py              # YOLOv8n-pose 감지 (NCNN / ONNX / PT 자동선택)
│   ├── pose.py                  # 스켈레톤 그리기
│   ├── annotate.py              # 한글 PIL 텍스트
│   ├── hado_movement.py         # 8동작 분류기 (ML + 규칙 기반)
│   ├── hado_movement_levels.py  # 게임 레벨별 동작 풀 (EASY/NORMAL/HARD)
│   ├── movement_demo.py         # 단순 동작 인식 데모 (게임 모드 X)
│   ├── rhythm_game.py           # ★ 게임 로직 (타이머/점수/타겟 선정)
│   ├── rhythm_game_ui.py        # ★ HUD/Intro/Result 렌더링
│   └── rhythm_game_demo.py      # ★ 게임 실행 메인
│
├── tests/
│   ├── test_hado_movement.py    # 8동작 분류 테스트
│   └── test_rhythm_game.py      # ★ 게임 로직 테스트 (10개 PASS)
│
├── tools/
│   ├── label_movements.py       # 라벨링 도구
│   └── train_movement_model.py  # 학습 스크립트
│
├── data/
│   └── reference_actions/labels.csv   # 202 샘플 라벨
│
├── models/
│   ├── hado_movement_clf.pkl          # 학습된 분류기
│   └── hado_movement_features.json
│
├── yolov8n-pose_ncnn_model/           # NCNN 최적화 모델 (Pi용)
├── yolov8n-pose.onnx                  # ONNX 모델
└── run.sh                             # 모든 명령어 진입점
```

---

## 🆙 다음 단계: 중급 동작 7개 추가 학습

내일(2026-06-19) 박진우님이 중급 동작 7개 사진을 추가 촬영해 학습할 예정.
구조가 **자동 확장 가능**하게 설계됨:

1. 새 사진을 `data/reference_actions/` 에 추가
2. `./run.sh label-movements` 로 라벨링 (새 라벨 7개)
3. `src/hado_movement_levels.py` 의 `INTERMEDIATE_MOVEMENTS` 리스트에 7개 이름 추가
4. `./run.sh train-model` 재학습
5. **게임 코드 수정 불필요** — `./run.sh game --level HARD` 만 하면 15개 동작 풀로 자동 동작

---

## 📅 개발 일정

- **Week 1–4**: 기존 HADO Smart Court 풀 시스템 (Level 1+2+3) — 완료 (별도 폴더 `hado-smart-court-iot/`)
- **Week 5 (6/18)**: 게임 모드 추가 (이 폴더), 발표 스코프 단순화
- **6/19**: 중급 동작 7개 학습 (HARD 모드)
- **6/20 (토)**: 🎤 최종발표 (녹화 영상 재생)

---

## 📝 라이선스

학술 프로젝트용. HADO는 meleap Inc.의 상표.
