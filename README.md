# Applying Schrödinger Bridge Methods to Generative Modeling of Financial Time Series

## Acknowledgment
This repository builds upon the code and methodology proposed by Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen in their paper "Generative Modeling for Time Series Via Schrödinger Bridge" (April 7, 2023). The original paper serves as the foundation for the implementation and experimentation carried out in this project.

## Introduction
This repository presents an implementation of the Schrödinger bridge (SB) approach for generative modeling of financial time series. The SB method offers a novel framework for modeling time series data, leveraging entropic interpolation via optimal transport to capture the temporal dynamics of financial markets. 

## Implementation Details
The code provided here is e.g. for financial time series data. It implements the SB approach described in the literature, offering functionalities to estimate the drift function from historical data samples and simulate synthetic time series data.

### Notebooks
- `onedim.ipynb`: Notebook for one-dimensional financial time series data, such as stock prices or index returns.
- `multidim.ipynb`: Notebook for multidimensional financial time series data, such as image sequences or high-dimensional market data.


## Performance Evaluation
The performance of the generative model is evaluated through various experiments on financial datasets, assessing metrics such as accuracy, robustness, and applicability to real-world scenarios.

## Future Work
This repository serves as a foundation for ongoing research in applying SB methods to financial time series. Future work includes refining the model, exploring alternative estimation techniques, and expanding the evaluation on diverse datasets.

## Master's Thesis
This project is part of a master's thesis titled "Applying Schrödinger Bridge Methods to Generative Modeling of Financial Time Series." The thesis aims to explore the effectiveness and practical implications of SB methods in financial data generation.

## Citation
If you find this code useful for your research, please consider citing the original paper that inspired this work:
Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen. "Generative Modeling for Time Series Via Schrödinger Bridge" (April 7, 2023). Available at SSRN: https://ssrn.com/abstract=4412434 or http://dx.doi.org/10.2139/ssrn.4412434

