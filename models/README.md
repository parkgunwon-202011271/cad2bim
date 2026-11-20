# ML 틈 판정 모델 (CAD2BIM_LLM=ml)

애매한 틈(문/창/벽/트임)을 LLM 대신 판정하는 RandomForest. 비용 0, 도면이 밖으로 나가지 않는다.

- 켜기: `pip install -e core[ml]` 후 `CAD2BIM_LLM=ml` (모델 경로는 config `review_ml.path`, 기본 `models/spot_model.joblib`)
- 모델: `spot_model.joblib` = 합성 100동(시드 1000~1099) + 물류 B동 판정 287곳으로 학습 (scikit-learn 1.9.1)
- 성적: 주민센터 24곳 평균 F1 0.73 (규칙 0.44, Gemini Flash Lite 0.60, GPT-5.5 0.92),
  합성 평가 30장 조립 결과 문 F1 0.871→0.891, 창 0.762→0.802
- 재학습: `python models/train_spot_model.py <특징 폴더> <출력.joblib>` (특징: `models/training/feat2_*.json`, `logistics.json`)
- 2026-11-20 밤 실험: 트임 보강 합성(feat3)·확장 합성(feat4)을 더하면 주민센터 0.71→0.50~0.54로 나빠져 쓰지 않았다
  (창 지점이 학습의 44%를 차지해 쏠림). 비율을 맞춰 뽑는 재학습이 다음 과제. 비교 스크립트: `final_eval.py`
