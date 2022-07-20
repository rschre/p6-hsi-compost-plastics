import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hsi_fullpath(data_folder):
    spectral.settings.envi_support_nonlowercase_params
    if os.path.exists(data_folder):
        capture_name = data_folder.split("\\")[-1]
        header_file = f"{data_folder}\capture\REFLECTANCE_{capture_name}.hdr"
        data_file = f"{data_folder}\capture\REFLECTANCE_{capture_name}.dat"
        datacube = spectral.io.envi.open(header_file, data_file)
        return datacube
    else:
        logger.error("Folder not found")
        return None


def get_hsi_capture(
    capture_name,
    data_folder="D:\\OneDriveFHNW\\FHNW\EUT-P6bb-21HS-RS_M365 - General\\captures",
):
    captures = glob.glob(f"{data_folder}\\*{capture_name}*")

    if len(captures) == 1:
        capture_name = captures[0].split("\\")[-1]
        header_file = f"{captures[0]}\\capture\\REFLECTANCE_{capture_name}.hdr"
        data_file = f"{captures[0]}\\capture\\REFLECTANCE_{capture_name}.dat"
        datacube = spectral.io.envi.open(header_file, data_file)
        logger.info(datacube)
        return datacube
    elif len(captures) > 0:
        logger.info(
            f"Multiple Captures matching pattern {capture_name}. Try being more precise."
        )
        logger.info(f"Found: {captures}")
        return None

    elif len(captures) == 0:
        logger.info(f"No matching capture found for {capture_name}")
        return None


def get_root_folder(device: str = None):

    root_pc = r"D:\OneDriveFHNW\FHNW\EUT-P6bb-21HS-RS_M365 - General\captures"
    root_notebook = r"C:\Users\Neu\FHNW\EUT-P6bb-21HS-RS_M365 - General\captures"

    if device == "PC":
        root = root_pc
    elif device == "NB":
        root = root_notebook
    else:
        select_root = input("Notebook / Home-PC? [NB/PC]")
        if select_root.upper() == "NB":
            root = root_notebook
        elif select_root.upper() == "PC":
            root = root_pc
        else:
            root = input("Enter alternative path for root:\n")

    logger.info(f"Root folder set to {root}")
    return root


def load_hsi_data(capture):
    return np.flip(capture.load(), axis=1)


def get_2d_roi(
    hsi_data: spectral.image.ImageArray,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
    band_low=8,
    band_high=210,
) -> np.ndarray:
    """
    Extracts a rectangular ROI of a given 3D-Array (HSI-Data) and returns it as a 2D-Array.

    :param hsi_data: 3D-Array of HSI-Data
    :param x_low: Leftmost Pixel of ROI on x-axis
    :param x_high: Rightmost Pixel of ROI on x-axis
    :param y_low: Top Pixel of ROI on y-axis
    :param y_high: Bottom Pixel of ROI on y-axis
    :param band_low: Lower wavelength bound of ROI
    :param band_high: Upper wavelength bound of ROI

    : return: 2D-Array of ROI
    """

    data = hsi_data[y_low:y_high, x_low:x_high, band_low:band_high]
    [m, n, p] = np.shape(data)
    data_2d = np.reshape(data, [m * n, p])

    return data_2d


def limit_reflection(data, threshold=8500):
    return np.where(data > threshold, threshold, data)


def display_roi_rectangle(
    hsi_data: spectral.image.ImageArray,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
    rgb_band_indexes=(81, 131, 181),
    title="Selected ROI",
):
    """
    Displays a rectangular ROI of the HSI-Data in a given figure.

    :param full_hsi_data: 3D-Array of HSI-Data
    :param x_low: Leftmost Pixel of ROI on x-axis
    :param x_high: Rightmost Pixel of ROI on x-axis
    :param y_low: Top Pixel of ROI on y-axis
    :param y_high: Bottom Pixel of ROI on y-axis
    :param rgb_band_indexes: Indexes of the hyperspectral bands for corresponding RGB-Bands
    :param title: Title of the figure

    :return: IMG Object
    """
    view_roi = spectral.imshow(
        np.where(hsi_data > 8500, 8500, hsi_data), rgb_band_indexes, title=title
    )
    view_roi.axes.add_patch(
        Rectangle(
            (x_low, y_low), x_high - x_low, y_high - y_low, fc="none", ec="r", lw=2
        )
    )

    return view_roi


def get_mean_plot(hsi_data_2d, column_names, title="Mean Spectrum", labels=None):
    df = pd.DataFrame(hsi_data_2d, columns=column_names)
    return df.mean().plot(labels=labels, title=title)


def snv_transform(input_data):
    """
    :snv: Standard Normal Variate Transformation:
    A correction technique which is done on each
    individual spectrum, a reference spectrum is not
    required. Subtracts the mean of every single spectrum (row) and divides
    by its standard deviation

    :param input_data: Array of spectral data, wavelengths as columns, pixels as rows
    :type input_data: DataFrame

    :returns: df_snv (DataFrame): Scatter corrected spectra
    """
    if isinstance(input_data, pd.DataFrame):
        cols = input_data.columns
        input_data = np.asarray(input_data)
        return_df = True
    else:
        return_df = False

    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    for i in range(data_snv.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(
            input_data[i, :]
        )

    if return_df:
        return pd.DataFrame(data_snv, columns=cols)
    else:
        return data_snv


def smooth_and_transform(
    data, smoothing_window_length=15, smoothing_polyorder=3, band_low=8, band_high=210
):
    """Smoothing the data using a Savitzy-Golay Filter and transforming the data to the range [0,1]
    Parameters
    ----------
    data : ndarray
        The data to be smoothed and transformed.
    smoothing_window_length : int
        The length of the smoothing window.
    smoothing_polyorder : int
        The order of the smoothing polynomial.
    band_low : int
        The lower bound of the band to be transformed.
    band_high : int
        The upper bound of the band to be transformed.
    Returns
    -------
    ndarray
        The smoothed and transformed data.
    """

    column_names = np.round(np.linspace(900, 1700, 224), 1)[band_low:band_high]
    data_smooth = savgol_filter(
        data,
        window_length=smoothing_window_length,
        polyorder=smoothing_polyorder,
        axis=1,
    )
    data_snv = snv_transform(data_smooth)
    data_df = pd.DataFrame(data_snv, columns=column_names)

    return data_df


def remove_outliers(hsi_data_2d, z_score_threshold=3, summary=False):
    """
    :param data_df: Dataframe of which outliers will be removed. pixels as rows, wavelength as columns
    :type data_df: DataFrame

    :param z_score_threshold: Threshold to be used to determine if a value is considered an outlier

    :returns: data_df (DataFrame): Dataframe with outliers removed

    """
    df_avg_zscore = pd.DataFrame(index=hsi_data_2d.index)
    z_score = (hsi_data_2d - hsi_data_2d.mean()) / hsi_data_2d.std()
    df_avg_zscore["z_score"] = z_score.abs().mean(axis=1)
    outliers = df_avg_zscore[df_avg_zscore["z_score"] > z_score_threshold].index

    if summary:
        print(f"Removed indexes: {list(outliers)}")

    return hsi_data_2d.drop(outliers)


def get_sample_plot(hsi_data: pd.DataFrame, sample_count: int, title=None, labels=None):
    """
    :param hsi_data: Dataframe of which a sample will be taken
    :param sample_count: Number of samples to be taken

    :returns: sample_plot: plotly plot of the sample
    """

    title = title if title else f"Sample of {sample_count} Spectra"
    sample_plot = hsi_data.sample(sample_count).T.plot.line(title=title, labels=labels)

    return sample_plot
