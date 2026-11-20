# 평가 재현 (발표 슬라이드 수치)

결과물: `results/ML_판정_비교.pdf`(발표 1장), `results/cad2bim_평가비교_발표용.xlsx`

준비 (저장소 맨 위): `pip install -e core[ml]`, DWG 변환기(ODA) 설치.

## ① 애매한 지점 판정: 우리 ML 대 LLM (주민센터 24곳)

- 대상: 주민센터 공개 도면에서 규칙이 스스로 정하지 못한 지점 24곳(`jumin_spots.json`).
  정답은 검토자 판정(트임 12, 문 6, 창 3, 벽 3).
- 우리: `models/spot_model.joblib`(RandomForest, 합성 100동 + 물류 B동 판정 287곳으로 학습).
  입력은 LLM에 주던 것과 같은 근거(지점과 겹치는 선의 레이어 역할별 개수, 양쪽 실, 가까운 글자, 폭).
- LLM: 같은 근거와 지점 이미지를 주고 문/창/벽/트임 중 하나를 물음. 답은 `results/llm_answers_jumin.txt`
  (로컬 Qwen3-VL 8B, Gemini 3.5 Flash Lite, GPT-5.5). 판정 시간은 이때 24곳을 실측한 값.
- 지표: 정확도, 4종 F1과 그 평균(평균 F1).

```
python eval/judgment_eval.py      # → results/judgment.json
```
근거는 평가 당시 LLM에 준 것을 `jumin_evidence.json`에 저장해 두었다. 도면을 새로 인식하면 지점 위치가 조금 달라져
같은 24곳이 나오지 않으므로, 이 파일로 판정만 다시 한다.

`judgment_eval.py`는 합성 시험 지점 378곳(학습에 안 쓴 다른 합성 30동)의 판정 F1도 함께 낸다.

## ② 평면도 문·창 개수: 우리 대 멀티모달 AI (합성 시험 도면 30장)

- 대상: 합성 생성기로 만든 10동, 평면 30장(시드 5000~5009, 개발·학습에 쓰지 않음). 평면 1장당 문 14개, 창 22개.
- 우리: 원본 CAD를 규칙+ML(`CAD2BIM_LLM=ml`)로 조립한 결과(`ir_built.json`)의 문·창 개수.
  비교용으로 규칙만(`CAD2BIM_LLM=off`)도 함께 잰다.
- Gemini: 같은 도면을 PNG로 그려 AECV-Bench와 같은 방식으로 문·창 개수를 물음.
- 지표(AECV-Bench와 같음): 정확 일치율(개수가 정답과 같은 평면 비율), MAPE(평균 오차율).
  위치 F1(정답 중심에서 max(600mm, 폭/2) 안, 같은 종류, 1:1 짝)도 함께 낸다.

```
sh eval/run_counts.sh _eval_work                                        # 생성 → 조립 → results/counts_ours.json
set GEMINI_API_KEY=...
python eval/vlm_counts.py _eval_work/synth_eval gemini:gemini-3.5-flash-lite   # → results/vlm/
```

- 논문 줄(Gemini 3 Pro, GPT-5.2, Claude Opus 4.5, Qwen3-VL 8B)은 직접 재지 않은 보고값:
  Kondratenko et al., *AECV-Bench*, arXiv:2601.04819 (2026). 주거 평면 120장(CubiCasa5K·CVC-FP·공개 도면)을
  PNG로 넣고 같은 프롬프트로 문·창·침실·화장실 개수를 물어 정확 일치율과 MAPE로 채점(애매한 경우 사람 판정).
  데이터가 우리 시험 도면과 다르므로 참고값이다.
