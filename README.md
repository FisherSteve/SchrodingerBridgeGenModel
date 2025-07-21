# Applying Schrödinger Bridge Methods to Generative Modeling of Time Series

## Acknowledgements
This repository builds on the code and methodology proposed by Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen in their paper "Generative Modelling for Time Series Via Schrödinger Bridge" (7 April 2023). The original paper and code serves as the basis for the implementation and experiments carried out in this project.

## Introduction
This repository presents an implementation of the Schrödinger Bridge (SB) approach to generative modelling of financial time series. The SB method provides a novel framework for modelling time series data, using entropic interpolation via optimal transport to capture, for example, the temporal dynamics of financial markets or to generate images.

## Requirements
The code was developed for **Python 3.10** and depends on the packages listed in
`requirements.txt`:

```
fbm==0.3.0
matplotlib==3.9.0
numpy==1.26.4
pandas==2.2.1
pandas-datareader==0.10.0
pybind11==2.9.1
scikit-learn==1.4.1
scipy==1.12.0
seaborn==0.13.2
statsmodels==0.14.0
tensorflow==2.16.1
tqdm==4.66.2
yfinance==0.2.40
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

It is recommended to create a virtual environment beforehand, e.g. using
`python3 -m venv sb_env` and `source sb_env/bin/activate`.

## Example

Here is an example of how an RGB image is generated using the SBTS.
These animations were generated using a step parameter for the initial part of the data (i.e. skipping every mth step), but no frames were skipped in the last 100 frames.


<table>
  <tr>
    <td><img src="./images/created_animation_step100_last100_1.gif" alt="Animation 1" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_2.gif" alt="Animation 2" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_3.gif" alt="Animation 3" width="200"/></td>
  </tr>
  <tr>
    <td><img src="./images/created_animation_step100_last100_4.gif" alt="Animation 4" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_6.gif" alt="Animation 6" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_9.gif" alt="Animation 9" width="200"/></td>
  </tr>
  <tr>
    <td><img src="./images/created_animation_step100_last100_10.gif" alt="Animation 10" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_12.gif" alt="Animation 12" width="200"/></td>
    <td><img src="./images/created_animation_step100_last100_14.gif" alt="Animation 14" width="200"/></td>
  </tr>
</table>

## Implementation details
The code provided here is for e.g. financial time series data. It implements the SB approach described in the literature and provides functionality to estimate the drift function from historical data samples and to simulate synthetic time series data.

### Notebooks
- `onedim.ipynb`: Notebook for one-dimensional financial time series data, such as stock prices or index returns.
- `multidim.ipynb`: Notebook for multidimensional financial time series data, such as image sequences or high-dimensional market data.

## Preprocessing price data
Use the `preprocess_prices` helper to transform raw price series into the array format expected by `SchrodingerBridge`.

```python
from src.preprocess import preprocess_prices

windows = preprocess_prices(data["Adj Close"], window_size=60)
bridge = SchrodingerBridge(distSize=60, nbpaths=windows.shape[0],
                           timeSeriesData=windows)
```


## Running the notebooks and scripts
1. Activate your environment and install the requirements as shown above.
2. Start Jupyter and open the desired notebook:

   ```bash
   jupyter notebook onedim.ipynb
   ```

   or

   ```bash
   jupyter notebook multidim.ipynb
   ```

   The one-dimensional notebook downloads Microsoft stock prices from Yahoo Finance using `yfinance`, while the multidimensional notebook loads image data from the `fashion_mnist` or `cifar10` datasets provided by TensorFlow.
3. After generating sample paths you can create GIF animations from the results:

   ```bash
   python animation_creation.py
   ```

   The script expects the data arrays created in the notebooks and saves the resulting GIFs to the `images/` directory.


## Performance evaluation
The performance of the generative model is evaluated through various experiments on financial datasets, assessing metrics such as accuracy, robustness and applicability to real-world scenarios.



## Master's Thesis
This project is part of a master's thesis titled "Applying Schrödinger Bridge Methods to Generative Modeling of Time Series" The thesis aims to explore the effectiveness and practical implications of SB methods in data generation.

## Citation
If you find this code useful for your research, please consider citing the original paper that inspired this work:
Hamdouche, Mohamed, Henry-Labordere, Pierre, and Pham, Huyen. "Generative Modeling for Time Series Via Schrödinger Bridge" (April 7, 2023). Available at SSRN: https://ssrn.com/abstract=4412434 or http://dx.doi.org/10.2139/ssrn.4412434


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

