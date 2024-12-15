
# Election Day Stock Market Forecasting (EDSMF)

## Overview

The **Election Day Stock Market Forecasting (EDSMF)** model aims to enhance stock market predictions during periods of high volatility, such as the United States presidential election. By integrating political signals, financial data, and advanced machine learning techniques, this project aims to capture both market dynamics and political influences, providing better forecasting accuracy for investors.

## Key Features

- **Political Signal Integration**: The model incorporates political signals generated from news articles about candidates' economic plans, enhancing prediction accuracy during election periods.
- **High-Frequency Trading Focus**: Unlike traditional models, EDSMF operates on a high-frequency trading setup with 1-minute interval data for all S&P 500 stocks during the U.S. presidential election period (30/10/2024 - 06/11/2024).
- **Granular Analysis**: The model captures detailed stock movement patterns by leveraging normalized stock indicators like open, high, low, close, volume, and Exponential Moving Average (EMA).
- **State-of-the-Art Forecasting**: Built on the \textit{StockMixer} (https://github.com/SJTU-DMTai/StockMixer) model, EDSMF combines established forecasting techniques with political awareness to predict market behavior with high accuracy.

## Installation

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/EDSMF.git
    cd EDSMF
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset:
    - The data is fetched through APIs or scraped from financial sources (details in the **Dataset** section).
    - Ensure that your environment has access to high-frequency stock data (1-minute interval for S&P 500 during election periods).

## Usage

1. Run the training script:
    ```bash
    python train_model.py
    ```

2. To test the model:
    ```bash
    python predict.py --data test_data.csv
    ```

3. For evaluation:
    ```bash
    python evaluate.py --model saved_model --data test_data.csv
    ```

## Dataset

- **Stock Data**: Historical stock data for all S&P 500 stocks is used with a 1-minute frequency between 30/10/2024 and 06/11/2024.
- **Political Data**: News articles, economic plans, and political developments related to the U.S. presidential election are processed using a large language model (LLM)-driven agent.

## Contributing

We welcome contributions! Feel free to fork the repository, open issues, or submit pull requests.

## License

This project is licensed under the MIT License.

---

### Acknowledgements

- \textit{StockMixer} framework for foundational stock forecasting techniques.
- Large Language Model (LLM)-driven agent for extracting political signals.
- Various contributors and open-source libraries that made this project possible.
