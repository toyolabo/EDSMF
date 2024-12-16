# Election Day Stock Market Forecasting (EDSMF)

## Main Reference

This project builds upon the excellent work of **StockMixer**. For more information, please visit the [StockMixer GitHub repository](https://github.com/SJTU-DMTai/StockMixer).

## Overview

The **Election Day Stock Market Forecasting (EDSMF)** model is designed to improve stock market predictions during high-volatility periods, such as the United States presidential election. By integrating **political signals**, **financial data**, and advanced **machine learning techniques**, EDSMF aims to capture the dynamic relationship between political events and market behavior, providing more accurate forecasts for investors during critical times.

Key features of the EDSMF model:
- **Political Signal Integration**: The model incorporates political signals based on news articles and candidate economic plans, providing insights into how political events impact stock movements.
- **High-Frequency Trading Focus**: Unlike traditional daily-frequency models, EDSMF operates on high-frequency stock data (1-minute intervals) for all S&P 500 stocks during the election period (30/10/2024 to 06/11/2024).
- **Granular Stock Analysis**: The model analyzes multiple stock indicators, including open, high, low, close, volume, and Exponential Moving Average (EMA), for improved prediction accuracy.
- **State-of-the-Art Forecasting**: Built upon the \textit{StockMixer} framework, EDSMF combines cutting-edge forecasting techniques with political awareness to predict stock behavior.

## Main Files

- `ensemble_train.py`: Main script to train the ensemble model.
- `sector_preprocess.ipynb`: Jupyter notebook for preprocessing sector data.

## Training Instructions

To train the **ensemble model**:

```bash
cd src
python run_ensemble_experiments.py
```

To train **random** and **baseline models**:

```bash
cd src
python run_experiment.py
```

## Running the Political Analyst

To execute the **political analyst** and generate political signals, use the following command:

```bash
cd political_analyst
crewai run
```

---

### Acknowledgements

- **StockMixer** for the foundational work in stock market forecasting.
- **CrewiAI** for political signal analysis.
- Various open-source libraries that made this project possible.
