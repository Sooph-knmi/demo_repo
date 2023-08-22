from typing import Dict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from aifs.diagnostics.logger import get_logger
from aifs.diagnostics.maps import Coastlines
from aifs.diagnostics.maps import EquirectangularProjection

LOGGER = get_logger(__name__)

continents = Coastlines()


def init_plot_settings():
    """Initialize matplotlib plot settings."""
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10

    mplstyle.use("fast")
    plt.rcParams["path.simplify_threshold"] = 0.9

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title


def _hide_axes_ticks(ax) -> None:
    """Hide x/y-axis ticks.

    Parameters
    ----------
    ax : _type_
        Axes object handle
    """

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
    """Create 2D plots of input, truth, and prediction.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axes object handle
    input_ : np.ndarray
        Input data
    truth : np.ndarray
        Expected data
    pred : np.ndarray
        Predicted data
    idx : int
        Sample index
    """
    cmap = cm.bwr
    cmap.set_bad(color="gray")

    pcm = ax[0].pcolormesh(truth, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[0].set_title(f"Truth idx={idx}")
    _hide_axes_ticks(ax[0])
    fig.colorbar(pcm, ax=ax[0])

    pcm = ax[1].pcolormesh(pred, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[1].set_title(f"Prediction idx={idx}")
    _hide_axes_ticks(ax[1])
    fig.colorbar(pcm, ax=ax[1])

    pcm = ax[2].pcolormesh(truth - pred, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[2].set_title("Prediction error")
    _hide_axes_ticks(ax[2])
    fig.colorbar(pcm, ax=ax[2])

    pcm = ax[3].pcolormesh(input_, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[3].set_title("Input")
    _hide_axes_ticks(ax[3])
    fig.colorbar(pcm, ax=ax[3])

    pcm = ax[4].pcolormesh(pred - input_, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[4].set_title("Increment [Pred - Input]")
    _hide_axes_ticks(ax[4])
    fig.colorbar(pcm, ax=ax[4])

    pcm = ax[5].pcolormesh(truth - input_, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0))
    ax[5].set_title("Persistence error")
    _hide_axes_ticks(ax[5])
    fig.colorbar(pcm, ax=ax[5])


def plot_predicted_multilevel_sample(
    n_plots_per_sample: int,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots data for one multilevel sample.

    Parameters
    ----------
    x : np.ndarray
        Input data (nvar*level, lat, lon)
    y_true : np.ndarray
        Expected data (nvar*level, lat, lon)
    y_pred : np.ndarray
        Predicted data (nvar*level, lat, lon)

    Returns
    -------
    Figure
        Figured object handle
    """
    n_plots_x, n_plots_y = y_true.shape[0], n_plots_per_sample

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

    Parameters
    ----------
    x : np.ndarray
        Data for Plotting of shape (npred,)

    Returns
    -------
    Figure
        The figure object handle.
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    colors = []
    for c in "krbgym":
        colors.extend([c] * 13)
    colors.extend(["c"] * 12)
    ax.bar(np.arange(x.size), x, color=colors, log=1)

    return fig


def plot_predicted_multilevel_flat_sample(
    parameters: Dict[str, int],
    n_plots_per_sample: int,
    latlons: np.ndarray,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots data for one multilevel latlon-"flat" sample.

    NB: this can be very slow for large data arrays
    call it as infrequently as possible!

    Parameters
    ----------
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    x : np.ndarray
        Input data of shape (lat*lon, nvar*level)
    y_true : np.ndarray
        Expected data of shape (lat*lon, nvar*level)
    y_pred : np.ndarray
        Predicted data of shape (lat*lon, nvar*level)

    Returns
    -------
    Figure
        The figure object handle.
    """
    n_plots_x, n_plots_y = len(parameters), n_plots_per_sample

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)

    pc = EquirectangularProjection()

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        xt = x[..., variable_idx].squeeze()
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()
        if n_plots_x > 1:
            plot_flat_sample(fig, ax[plot_idx, :], pc, latlons, xt, yt, yp, variable_name)
        else:
            plot_flat_sample(fig, ax, pc, latlons, xt, yt, yp, variable_name)
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
    """Plot a "flat" 1D sample.

    Data on non-rectangular (reduced Gaussian) grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axis object handle
    pc : _type_
        CRS, eg. ccrs.PlateCarreee
    latlons : np.ndarray
        lat/lon coordinates array, shape (lat*lon, 2)
    input_ : np.ndarray
        Input data of shape (lat*lon,)
    truth : np.ndarray
        Expected data of shape (lat*lon,)
    pred : np.ndarray
        Predicted data of shape (lat*lon,)
    vname : str
        Variable name
    """

    lat, lon = latlons[:, 0], latlons[:, 1]

    scatter_plot(fig, ax[0], pc, lat, lon, input_, title=f"{vname} input")
    scatter_plot(fig, ax[1], pc, lat, lon, truth, title=f"{vname} target")
    scatter_plot(fig, ax[2], pc, lat, lon, pred, title=f"{vname} pred")
    scatter_plot(fig, ax[3], pc, lat, lon, truth - pred, cmap="bwr", title=f"{vname} pred err")
    scatter_plot(fig, ax[4], pc, lat, lon, pred - input_, cmap="bwr", title=f"{vname} increment [pred - input]")
    scatter_plot(fig, ax[5], pc, lat, lon, truth - input_, cmap="bwr", title=f"{vname} persist err")


def scatter_plot(
    fig, ax, pc, lat: np.array, lon: np.array, data: np.array, cmap: str = "viridis", title: Optional[str] = None
) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axis object handle
    pc : _type_
        CRS, eg. ccrs.PlateCarreee
    lat : _type_
        Latitudes
    lon : _type_
        Longitudes
    data : _type_
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    title : _type_, optional
        Title for plot, by default None
    """

    psc = ax.scatter(
        *pc(lon, lat),
        c=data,
        cmap=cmap,
        s=1,
        alpha=1.0,
        norm=TwoSlopeNorm(vcenter=0.0) if cmap == "bwr" else None,
    )
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi / 2, np.pi / 2))

    continents.plot_continents(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
    plt.tight_layout()
    fig.colorbar(psc, ax=ax)


def plot_graph_features(
    latlons: np.ndarray,
    features: np.ndarray,
) -> Figure:
    """Plot trainable graph features.

    Parameters
    ----------
    latlons : np.ndarray
        Latitudes and longitudes
    features : np.ndarray
        Trainable Features

    Returns
    -------
    Figure
        Figure object handle
    """
    nplots = features.shape[-1]
    figsize = (nplots * 4, 3)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    lat, lon = latlons[:, 0], latlons[:, 1]

    pc = EquirectangularProjection()
    for i in range(nplots):
        ax_ = ax[i] if nplots > 1 else ax
        scatter_plot(fig, ax_, pc, lat, lon, features[..., i])
    return fig


def plot_predicted_ensemble() -> None:
    raise NotImplementedError
