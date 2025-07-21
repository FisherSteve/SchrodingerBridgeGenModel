import numpy as np
import pandas as pd


def preprocess_prices(prices, window_size, log_returns=True):
    """Return time series windows for :class:`SchrodingerBridge`.

    Parameters
    ----------
    prices : pandas.Series, pandas.DataFrame or array-like
        Raw price observations. If a ``DataFrame`` is provided, all columns
        will be processed.
    window_size : int
        Length of each rolling window (``N``).
    log_returns : bool, optional
        If ``True`` (default), compute log returns of the normalised prices.
        When ``False`` the normalised price paths are returned instead.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(M, window_size + 1, D)`` with ``M`` windows and
        ``D`` features. When ``D`` equals 1 the trailing dimension is
        squeezed. The first column is zero when ``log_returns`` is ``True``.

    Examples
    --------
    >>> import yfinance as yf
    >>> from src.preprocess import preprocess_prices
    >>> data = yf.download("MSFT", start="2010-01-01", end="2020-01-30")
    >>> windows = preprocess_prices(data["Adj Close"], window_size=60)
    >>> sb = SchrodingerBridge(distSize=60, nbpaths=windows.shape[0],
    ...                        timeSeriesData=windows)
    """
    arr = np.asarray(prices)
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        arr = prices.values
    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.shape[0] < window_size + 1:
        raise ValueError("window_size larger than number of observations")

    # Build rolling windows
    view = np.lib.stride_tricks.sliding_window_view(arr, window_size + 1, axis=0)
    windows = view.reshape(-1, window_size + 1, arr.shape[1])

    # Normalise by first value in each window
    norm = windows / windows[:, :1, :]

    if log_returns:
        log_ret = np.zeros_like(norm)
        log_ret[:, 1:, :] = np.diff(np.log(norm), axis=1)
        result = log_ret
    else:
        result = norm

    if result.shape[2] == 1:
        result = result[:, :, 0]

    return result
