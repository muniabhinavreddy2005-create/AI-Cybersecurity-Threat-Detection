# Architecture & Data Flow

```
+--------------------+     +------------------+     +----------------------+
| Data source        | --> | Cleaning         | --> | Feature engineering  |
| simulate.py /      |     | dedupe, inf->NaN |     | total_bytes, ratios, |
| data_loader.py     |     | (preprocess.py)  |     | failed_login_rate,   |
+--------------------+     +------------------+     | is_night             |
                                                    +----------+-----------+
                                                               |
                          +------------------------------------v------------------+
                          | Preprocessing pipeline: impute -> scale -> one-hot    |
                          +-----------+----------------------------------+--------+
                                      |                                  |
                         +------------v-----------+        +-------------v-----------+
                         | Supervised models      |        | Isolation Forest        |
                         | RF + LogReg (train.py) |        | trained on NORMAL only  |
                         +------------+-----------+        +-------------+-----------+
                                      |                                  |
                                +-----v----------------------------------v-----+
                                | detect.py: probability -> severity + reason   |
                                +-----+---------------------------+-------------+
                                      |                           |
                          +-----------v---------+      +----------v-----------+
                          | outputs/alerts/*.csv|      | visualize.py -> images/ |
                          +---------------------+      +-------------------------+
                                      |
                                +-----v------------+
                                | predict_api.py   |  real-time /predict
                                +------------------+
```

| Module | Responsibility |
|---|---|
| `simulate.py` | Generates traffic for 5 behaviours, adds noise, stealth attacks, hard negatives |
| `data_loader.py` | Loads simulated data or NSL-KDD |
| `preprocess.py` | Cleaning, feature engineering, pipeline, leakage-safe X/y split |
| `train.py` | Stratified split, trains 2 classifiers + Isolation Forest, saves models |
| `evaluate.py` | Accuracy, precision, recall, F1, confusion matrix |
| `detect.py` | Severity bands, human-readable reasons, alert CSV |
| `visualize.py` | 7 portfolio-ready charts |
| `predict_api.py` | Flask endpoint for single-flow prediction |
