# Interview Prep (answers match THIS code)

1. **Why simulated data?** No access to corporate logs; simulation lets me control attack types and show the pipeline end-to-end. I also made it loadable with NSL-KDD for real data.
2. **Why not just report accuracy?** ~88% of traffic is normal, so a model that says "normal" always scores 88%. I use precision, recall, F1 and the confusion matrix.
3. **Why class_weight="balanced" and stratified split?** Keeps rare attacks from being ignored and keeps the same ratio in train/test.
4. **Isolation Forest vs Random Forest?** RF needs labelled attacks and is more precise; Isolation Forest is trained on normal-only traffic and can catch unseen attack types but has more false alarms (precision 0.69 in my run).
5. **What is data leakage and how did you avoid it?** Labels (`label`, `is_attack`) and IDs (`flow_id`, `src_ip`) are removed from features; preprocessing is inside the Pipeline so scaling is fit only on training data.
6. **How do you reduce false positives?** Raise the probability threshold, use severity bands, combine RF + anomaly score, add analyst feedback loop.
7. **Limitations?** Simulated data is cleaner than real traffic; no streaming; static model drifts over time.
8. **Next steps?** CICIDS2017, SHAP explanations, Streamlit dashboard, Docker, Kafka streaming, scheduled retraining.
