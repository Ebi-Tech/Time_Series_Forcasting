# Comparing Sequential Models for Forecasting Mobile Network Traffic

A case study on the Milan telecommunications activity dataset: comparing a classical statistical model (SARIMA), a recurrent neural network (LSTM), and a gradient-boosted tree model (LightGBM) on one-step-ahead Internet traffic forecasting.

**Author:** Divine Ebube Ifechukwude

---

## Table of contents

- [Overview](#overview)
- [Research question](#research-question)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Pipeline summary](#pipeline-summary)
- [Results summary](#results-summary)
- [Key findings](#key-findings)
- [Limitations and future work](#limitations-and-future-work)
- [Report and video](#report-and-video)
- [References](#references)

---

## Overview

This project forecasts Internet traffic activity for small geographic areas in the city of Milan, using historical mobile network data collected over roughly two months in late 2013. Three models, each representing time and traffic in a fundamentally different way, are built and compared on the same forecasting problem:

| Model | Type | Framework |
|---|---|---|
| SARIMA | Classical statistical, seasonal | `statsmodels` |
| LSTM | Recurrent neural network | `tensorflow` / `keras` |
| LightGBM | Gradient-boosted trees on engineered features | `lightgbm` |

The goal is not just to find the model with the lowest error, but to understand *why* one model performs better or worse in a given area, tying every result back to what the underlying data actually looks like.

## Research question

*How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?*

## Dataset

The data comes from a public dataset released by Telecom Italia, hosted on Harvard Dataverse:

- **Source:** [Harvard Dataverse, DOI: 10.7910/DVN/EGZHFV](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
- **Coverage:** City of Milan and surroundings, split into a grid of 10,000 square areas
- **Time span:** November 1, 2013 to January 1, 2014 (62 daily files)
- **Resolution:** 10-minute intervals
- **Size:** Roughly 20 GB combined across all 62 files
- **Columns per row:** `square_id`, `timestamp`, `country_code`, `sms_in`, `sms_out`, `call_in`, `call_out`, `internet`

Only the `internet` column (Internet traffic) is used for analysis and forecasting in this project. The other activity columns exist in the raw files but were dropped once it was confirmed they were not needed.

### A note on downloading the data

The dataset is not available as a single download. Each of the 62 daily files must be requested individually through Dataverse's file access API, and Dataverse requires a short guestbook submission (name, email, institution) before releasing a file. Since this cannot practically be done by hand 62 times, the download is fully scripted:

1. A `POST` request is sent to `https://dataverse.harvard.edu/api/access/datafile/{file_id}` with a JSON body containing the guestbook response.
2. Dataverse responds with a short-lived, one-time **signed URL**.
3. That signed URL is immediately `GET`-requested to retrieve the actual file.

A plain `requests.get()` to Dataverse's API (even for metadata) can return an HTTP 403 on some environments (Kaggle included) unless a browser-like `User-Agent` header is set on every request. This is handled in Notebook 1; see that notebook for the exact working code.

No manual downloading, browser interaction, or guestbook form-filling is required to reproduce this pipeline, everything is scripted end to end.

## Repository structure

```
.
├── notebooks/
│   ├── 01_ingestion_and_eda.ipynb        
│   ├── 02_model_sarima.ipynb             
│   ├── 03_model_lstm.ipynb              
│   ├── 04_model_lightgbm.ipynb           
│   └── 05_final_comparison.ipynb         
├── report/
│   └── Traffic_Forecasting_Report.docx  
├── requirements.txt
└── README.md
```

Each model notebook (02-04) is fully **independent**: it only reads the shared data produced by Notebook 1, never another model notebook's output. All three models' results are only combined together in Notebook 5. This keeps the comparison fair, since no model's training or evaluation is ever influenced by another model's results.

## Setup

This project was built and run on **Kaggle Notebooks**, since the dataset (~20 GB combined) and the LSTM's GPU requirement are impractical on a typical laptop. It can also be adapted to run locally or on another cloud notebook environment, provided the dependencies below are installed and enough disk space is available for temporary file downloads.

### Dependencies

```
pandas
numpy
matplotlib
scikit-learn
statsmodels
tensorflow
lightgbm
requests
```

Install with:

```bash
pip install -r requirements.txt
```

### Hardware used per notebook

| Notebook | Accelerator | Why |
|---|---|---|
| 01 Ingestion & EDA | None (CPU) | Only downloading, aggregating, and plotting, no model training |
| 02 SARIMA | None (CPU) | Classical statistical model, no GPU-accelerated path exists |
| 03 LSTM | GPU (Tesla T4 x2 on Kaggle) | Neural network training benefits meaningfully from GPU |
| 04 LightGBM | None (CPU) | Tree-based model is already fast and efficient on CPU |
| 05 Final comparison | None (CPU) | Only loads small saved CSV files and builds plots/tables |

## How to run

Run the notebooks **in order**. Each one saves its outputs, which the next notebook needs.

1. **`01_ingestion_and_eda.ipynb`**
   Downloads and processes the raw dataset, builds the exploratory analysis, and produces the cleaned per-area time series files used by every model. No inputs need to be attached; this notebook only needs internet access enabled (on Kaggle: toggle "Internet" on in notebook settings).

   **Outputs:** `file_manifest.csv`, `square_traffic_totals.csv`, `five_areas_timeseries.csv`, `clean_series_5161.csv`, `clean_series_5059.csv`, `clean_series_5259.csv`

   > When saving this notebook's version, make sure **"Save outputs"** is checked, otherwise the generated CSV files will not persist and will not be available to the next notebooks.

2. **`02_model_sarima.ipynb`**, **`03_model_lstm.ipynb`**, **`04_model_lightgbm.ipynb`**
   Each of these needs Notebook 1's output attached as an input (on Kaggle: "Add Input" → search for Notebook 1 by name). These three can be run in any order relative to each other, since they do not depend on one another.

   **Outputs (per notebook):** `<model>_predictions.csv`, `<model>_metrics.csv`, `<model>_timing.csv`, `<model>_hourly_error.csv`, plus forecast and error plots. LightGBM's notebook additionally outputs `lgbm_tuning_results.csv` and `lgbm_tuning_comparison.csv`.

   Save each notebook's version with outputs included, same as Notebook 1.

3. **`05_final_comparison.ipynb`**
   Needs Notebooks 2, 3, and 4 all attached as inputs. Combines all three models' saved metrics and predictions into the final comparison tables, the nine required forecast plots, and the comparative discussion, including a dedicated failure analysis section.

## Pipeline summary

```
Raw data (62 files, ~20GB, Harvard Dataverse)
        │
        ▼
Notebook 1: Ingestion & EDA
  ├─ Pass 1: stream all 62 files, compute per-area traffic totals
  │          → identifies busiest 3 areas: squares 5161, 5059, 5259
  ├─ Pass 2: stream all 62 files again, extract 5 areas of interest
  │          (top-3 in full, plus squares 4159 & 4556 for 2 weeks only)
  ├─ Exploratory analysis: distribution, temporal comparison,
  │          autocorrelation, anomaly detection
  └─ Shared preprocessing: one clean, gap-free file per top-3 area,
             with a fixed Dec 16-22 test window
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
  Notebook 2       Notebook 3     Notebook 4
   (SARIMA)          (LSTM)       (LightGBM)
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
              Notebook 5: Final comparison
        (combined tables, 9 plots, discussion, failure analysis)
```

Only Internet traffic (`internet` column) is carried through to the final modelling stage. SMS and call activity are present in the raw files and were kept through the extraction step in case they turned out useful, but were dropped once it was confirmed only Internet traffic was needed for this study.

## Results summary

One-step-ahead forecasting performance, test week of December 16-22, 2013:

**Square 5161**

| Model | MAE | MAPE | RMSE |
|---|---|---|---|
| SARIMA | 103.76 | 12.25% | 149.40 |
| LSTM | 108.60 | 12.98% | 152.93 |
| LightGBM | 82.31 | 8.57% | 122.75 |

**Square 5059**

| Model | MAE | MAPE | RMSE |
|---|---|---|---|
| SARIMA | 100.32 | 11.57% | 151.86 |
| LSTM | 102.34 | 9.59% | 141.05 |
| LightGBM | 71.72 | 7.21% | 101.02 |

**Square 5259**

| Model | MAE | MAPE | RMSE |
|---|---|---|---|
| SARIMA | 89.18 | 12.35% | 124.86 |
| LSTM | 78.21 | 8.06% | 116.996 |
| LightGBM | 66.27 | 7.30% | 96.66 |

**Training / inference time per area (approximate):**

| Model | Training | Inference (full test week) | Hardware |
|---|---|---|---|
| SARIMA | < 1 second | ~50 seconds | CPU only |
| LSTM | 26-46 seconds | ~0.3 seconds | GPU (Tesla T4 x2) |
| LightGBM | 0.16-0.25 seconds | 0.006-0.009 seconds | CPU only |

LightGBM achieved the lowest error on every metric across all three areas, while also being the fastest to train and to use for prediction.

## Key findings

- **LightGBM won consistently**, attributed to it being the only model given an explicit, direct way to represent both the regular daily traffic pattern and specific known holiday dates at the same time.
- **SARIMA** correctly captured the dominant 24-hour seasonal pattern but showed a clear, one-off instability at the very start of the test week, and has no way to represent the two-directional (sometimes higher, sometimes lower) holiday effects found in the data.
- **LSTM** showed a genuine, systematic underprediction bias specific to square 5059, present at every hour of the day, a real weakness worth noting despite this model's overall flexibility.
- **A visual assumption was checked and corrected**: square 5259's visually "rougher" December 21-22 period was assumed to be harder to predict, but measuring it directly showed *lower* error for all three models during that period, since it also involved lower overall traffic magnitude. This led to a general finding: MAE and RMSE scale with traffic magnitude, while MAPE is a more reliable measure of relative predictability across periods of differing traffic levels.
- All three models' errors concentrated around the same time of day (roughly 11:00-17:00, the sharpest part of the daily traffic swing), suggesting this reflects a genuine difficulty in the forecasting problem itself, not a weakness specific to any one architecture.

Full reasoning, evidence, and discussion for every finding above is in the written report and in the notebooks themselves (each step includes a markdown cell explaining the reasoning before the code, and a markdown cell interpreting the real output after it).

## Limitations and future work

- All tuning and modelling was done on the three busiest areas only; it is not yet known whether the same pattern (LightGBM ahead, SARIMA and LSTM each with their own specific weaknesses) holds for quieter areas.
- The LSTM's underprediction bias on square 5059 was identified but not resolved; a different architecture, more training data, or a different scaling approach could be tried.
- Combining approaches, for example feeding LightGBM's engineered features into a model that also retains some memory of recent history, is a natural next step suggested by the literature reviewed in this project.

## Report and video

- Full written report: [`report/Traffic_Forecasting_Report.docx`](./report/Traffic_Forecasting_Report.docx)
- Video presentation: *[link to be added]*

## References

[1] G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli, A. Vespignani, A. Pentland, and B. Lepri, "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," *Sci. Data*, vol. 2, art. no. 150055, 2015, doi: 10.1038/sdata.2015.55.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular traffic prediction and classification: a comparative evaluation of LSTM and ARIMA," in *Discovery Science*, Split, Croatia, 2019, pp. 129-144, doi: 10.1007/978-3-030-33778-0_11.

[3] O. Aouedi, V. Le, K. Piamrat, and Y. Ji, "A survey on deep learning for cellular traffic prediction," *Intell. Comput.*, 2024, doi: 10.34133/icomputing.0054.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.

[5] B. N. Oreshkin et al., "Forecasting with gradient boosted trees: augmentation, tuning, and cross-validation strategies," *Int. J. Forecast.*, 2022, doi: 10.1016/j.ijforecast.2021.10.004.