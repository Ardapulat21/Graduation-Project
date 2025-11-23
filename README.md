# Time Series Forecasting Pipeline: MSTL, LOF, and ARIMA

# Project Overview

This repository contains the source code and documentation for my Graduation Project. The core objective of this study is to enhance time series forecasting accuracy by implementing a robust preprocessing and decomposition pipeline.

Traditional forecasting models often struggle with noise and complex seasonal patterns. To address this, this project implements a hybrid approach:

1.  **LOF (Local Outlier Factor)** for anomaly detection and data cleaning.
2.  **MSTL (Multiple Seasonal-Trend decomposition using Loess)** for structural decomposition.
3.  **ARIMA (AutoRegressive Integrated Moving Average)** for the final forecasting modeling.

## Methodology

The project follows a strict data science pipeline designed to isolate signal from noise before attempting predictions.

### 1. Data Preprocessing & Anomaly Detection (LOF)

Raw time series data often contains anomalies that can skew statistical properties. I utilized **Local Outlier Factor (LOF)**, a density-based algorithm, to identify and handle these outliers.

* **Why LOF?** Unlike simple statistical thresholding, LOF detects local outliers by comparing the density of an instance to the density of its neighbors, making it effective for variable time series data.

<img width="1044" height="441" alt="LOF" src="https://github.com/user-attachments/assets/8414d644-365c-4d54-894c-c850ad90ad77" />


### 2. Time Series Decomposition (MSTL)

After cleaning the data, I applied **MSTL** to break the time series down into its constituent components.

* **Input:** A dataset exhibiting specific seasonal patterns.
* **Process:** MSTL separates the series into:
    * **Trend:** The long-term progression of the series.
    * **Seasonal:** The repeating short-term cycle.
    * **Residual:** The remaining random noise.
* **Significance:** By isolating the seasonality, we can model the underlying trend more accurately without the interference of cyclic fluctuations.


<img width="1287" height="496" alt="MSTL" src="https://github.com/user-attachments/assets/6940c168-589b-41e2-b1a5-c8978888c5b9" />

### 3. Forecasting (ARIMA)

With the series decomposed and the noise isolated, **ARIMA** is applied to model the components.

<img width="1063" height="436" alt="ARIMA" src="https://github.com/user-attachments/assets/2b59fc39-eb0f-4c64-84a1-343a22db3219" />

* The model parameters $(p, d, q)$ are optimized to fit the smoothed trend and seasonal data, resulting in a more stable and accurate forecast compared to running ARIMA on raw, noisy data.
