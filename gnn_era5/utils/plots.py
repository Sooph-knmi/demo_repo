import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure

import cartopy.crs as ccrs
import cartopy.feature as cf

from gnn_era5.utils.logger import get_logger
from gnn_era5.utils.constants import _NUM_VARS_TO_PLOT, _NUM_PLOTS_PER_SAMPLE, _IDXVARS_TO_PLOT, _NAM_VARS_TO_PLOT

LOGGER = get_logger(__name__)


def init_plot_settings():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title


def _hide_axes_ticks(ax) -> None:
    # hide x/y-axis ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)


def plot_2d_sample(
    fig,
    ax,
    input_: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    idx: int,
) -> None:
    cmap = cm.bwr
    cmap.set_bad(color="gray")

    pcm = ax[0].pcolormesh(truth, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[0].set_title(f"Truth idx={idx}")
    _hide_axes_ticks(ax[0])
    fig.colorbar(pcm, ax=ax[0])

    pcm = ax[1].pcolormesh(pred, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[1].set_title(f"Prediction idx={idx}")
    _hide_axes_ticks(ax[1])
    fig.colorbar(pcm, ax=ax[1])

    pcm = ax[2].pcolormesh(truth - pred, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[2].set_title("Prediction error")
    _hide_axes_ticks(ax[2])
    fig.colorbar(pcm, ax=ax[2])

    pcm = ax[3].pcolormesh(input_, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[3].set_title("Input")
    _hide_axes_ticks(ax[3])
    fig.colorbar(pcm, ax=ax[3])

    pcm = ax[4].pcolormesh(pred - input_, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[4].set_title("Increment [Pred - Input]")
    _hide_axes_ticks(ax[4])
    fig.colorbar(pcm, ax=ax[4])

    pcm = ax[5].pcolormesh(truth - input_, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[5].set_title("Persistence error")
    _hide_axes_ticks(ax[5])
    fig.colorbar(pcm, ax=ax[5])


def plot_2d(
    fig,
    ax,
    array: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    idx: int,
) -> None:
    cmap = cm.bwr
    cmap.set_bad(color="gray")

    pcm = ax[0].pcolormesh(truth, cmap=cmap, norm=colors.TwoSlopeNorm(vcenter=0.0))
    ax[0].set_title(f"Loss idx={idx}")
    _hide_axes_ticks(ax[0])
    fig.colorbar(pcm, ax=ax[0])


def plot_predicted_multilevel_sample(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots data for one multilevel sample.
    Args:
        x, y_true, y_pred: arrays of shape (nvar*level, lat, lon)
    Returns:
        The figure object handle.
    """
    n_plots_x, n_plots_y = y_true.shape[0], _NUM_PLOTS_PER_SAMPLE

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    for idx in range(n_plots_x):
        xt = x[idx, ...].squeeze()
        yt = y_true[idx, ...].squeeze()
        yp = y_pred[idx, ...].squeeze()
        if n_plots_x > 1:
            plot_2d_sample(fig, ax[idx, :], xt, yt, yp, idx)
        else:
            plot_2d_sample(fig, ax, xt, yt, yp, idx)
    return fig


def plot_loss(
    x: np.ndarray,
) -> Figure:
    """Plots data for one multilevel sample.
    Args:
        x arrays of shape (npred,)
    Returns:
        The figure object handle.
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    colors = []
    for c in "krbgym":
        colors.extend([c] * 13)
    colors.extend(["c"] * 12)
    ax.bar(np.arange(x.size), x, color=colors, log=1)

    return fig


# ---------------------------------------------------------------
# NB: this can be very slow for large data arrays
# call it as infrequently as possible!
# ---------------------------------------------------------------
def plot_predicted_multilevel_flat_sample(
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots data for one multilevel latlon-"flat" sample.
    Args:
        latlons: lat/lon coordinates array, shape (lat*lon, 2)
        x, y_true, y_pred: arrays of shape (lat*lon, nvar*level)
    Returns:
        The figure object handle.
    """
    n_plots_x, n_plots_y = _NUM_VARS_TO_PLOT, _NUM_PLOTS_PER_SAMPLE

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    pc = ccrs.PlateCarree()

    for vix, idx in enumerate(_IDXVARS_TO_PLOT):
        vname = _NAM_VARS_TO_PLOT[vix]
        xt = x[..., idx].squeeze()
        yt = y_true[..., idx].squeeze()
        yp = y_pred[..., idx].squeeze()
        if n_plots_x > 1:
            plot_flat_sample(fig, ax[vix, :], pc, latlons, xt, yt, yp, vname)
        else:
            plot_flat_sample(fig, ax, pc, latlons, xt, yt, yp, vname)
    return fig


def plot_flat_sample(
    fig,
    ax,
    pc,
    latlons: np.ndarray,
    input_: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    vname: str,
) -> None:
    """Use this with `flat` (1D) samples, e.g. data on non-rectangular (reduced Gaussian) grids."""

    lat, lon = latlons[:, 0], latlons[:, 1]

    scatter_plot(fig, ax[0], pc, lat, lon, truth, title=f"{vname} target")
    scatter_plot(fig, ax[1], pc, lat, lon, pred, title=f"{vname} pred")
    scatter_plot(fig, ax[2], pc, lat, lon, truth - pred, title=f"{vname} pred err")
    scatter_plot(fig, ax[3], pc, lat, lon, input_, title=f"{vname} input")
    scatter_plot(fig, ax[4], pc, lat, lon, pred - input_, title=f"{vname} increment [pred - input]")
    scatter_plot(fig, ax[5], pc, lat, lon, truth - input_, title=f"{vname} persist err")


def scatter_plot(fig, ax, pc, lat, lon, x, title=None) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids."""
    ax.set_global()
    ax.add_feature(cf.COASTLINE, edgecolor="black", linewidth=0.5)

    psc = ax.scatter(
        x=lon,
        y=lat,
        c=x,
        cmap="bwr",
        s=1,
        alpha=1.0,
        transform=pc,
        norm=colors.TwoSlopeNorm(vcenter=0.0),
    )
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("auto", adjustable=None)
    fig.colorbar(psc, ax=ax)
