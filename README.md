# Applying Schrödinger Bridge Methods to Generative Modeling of Financial Time Series

## Acknowledgment
This repository builds upon the code and methodology proposed by Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen in their paper "Generative Modeling for Time Series Via Schrödinger Bridge" (April 7, 2023). The original paper serves as the foundation for the implementation and experimentation carried out in this project.

## Introduction
This repository presents an implementation of the Schrödinger bridge (SB) approach for generative modeling of financial time series. The SB method offers a novel framework for modeling time series data, leveraging entropic interpolation via optimal transport to capture the temporal dynamics of financial markets. 

## Example

Here is a small example on how an RGB image is being generated with the SBTS
These animations were generated using different step parameters for the initial part of the data (i.e. skipping every m-th step), but no frames were skipped in the last 100 frames.

<table>
  <tr>
    <td><img src="./images/created_animation_step100_last100_1.gif" alt="Animation 1" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_2.gif" alt="Animation 2" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_3.gif" alt="Animation 3" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_4.gif" alt="Animation 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="./images/created_animation_step100_last100_5.gif" alt="Animation 5" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_6.gif" alt="Animation 6" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_7.gif" alt="Animation 7" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_8.gif" alt="Animation 8" width="200"/></td>
  </tr>
  <tr>
    <td><img src="./images/created_animation_step100_last100_9.gif" alt="Animation 9" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_10.gif" alt="Animation 10" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_11.gif" alt="Animation 11" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_12.gif" alt="Animation 12" width="200"/></td>
  </tr>
  <tr>
    <td><img src="./images/created_animation_step100_last100_13.gif" alt="Animation 13" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_14.gif" alt="Animation 14" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_15.gif" alt="Animation 15" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_16.gif" alt="Animation 16" width="200"/></td>
  </tr>
</table>






## Implementation Details
The code provided here is e.g. for financial time series data. It implements the SB approach described in the literature, offering functionalities to estimate the drift function from historical data samples and simulate synthetic time series data.

### Notebooks
- `onedim.ipynb`: Notebook for one-dimensional financial time series data, such as stock prices or index returns.
- `multidim.ipynb`: Notebook for multidimensional financial time series data, such as image sequences or high-dimensional market data.


## Performance Evaluation
The performance of the generative model is evaluated through various experiments on financial datasets, assessing metrics such as accuracy, robustness, and applicability to real-world scenarios.


## Master's Thesis
This project is part of a master's thesis titled "Applying Schrödinger Bridge Methods to Generative Modeling of Financial Time Series." The thesis aims to explore the effectiveness and practical implications of SB methods in financial data generation.

## Citation
If you find this code useful for your research, please consider citing the original paper that inspired this work:
Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen. "Generative Modeling for Time Series Via Schrödinger Bridge" (April 7, 2023). Available at SSRN: https://ssrn.com/abstract=4412434 or http://dx.doi.org/10.2139/ssrn.4412434

