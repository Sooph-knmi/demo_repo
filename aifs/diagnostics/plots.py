from typing import Dict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from aifs.diagnostics.maps import Coastlines
from aifs.diagnostics.maps import EquirectangularProjection
from aifs.utils.logger import get_code_logger


LOGGER = get_code_logger(__name__)

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
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)

    for plot_idx, (variable_idx, (variable_name, output_only)) in enumerate(parameters.items()):
        xt = x[..., variable_idx].squeeze() * int(output_only)
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()
        if n_plots_x > 1:
            plot_flat_sample(fig, ax[plot_idx, :], pc_lon, pc_lat, xt, yt, yp, variable_name)
        else:
            plot_flat_sample(fig, ax, pc_lon, pc_lat, xt, yt, yp, variable_name)

    return fig


def plot_flat_sample(
    fig,
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
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
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    input_ : np.ndarray
        Input data of shape (lat*lon,)
    truth : np.ndarray
        Expected data of shape (lat*lon,)
    pred : np.ndarray
        Predicted data of shape (lat*lon,)
    vname : str
        Variable name
    """

    scatter_plot(fig, ax[1], lon, lat, truth, title=f"{vname} target")
    scatter_plot(fig, ax[2], lon, lat, pred, title=f"{vname} pred")
    scatter_plot(fig, ax[3], lon, lat, truth - pred, cmap="bwr", title=f"{vname} pred err")
    if sum(input_) != 0:
        scatter_plot(fig, ax[0], lon, lat, input_, title=f"{vname} input")
        scatter_plot(fig, ax[4], lon, lat, pred - input_, cmap="bwr", title=f"{vname} increment [pred - input]")
        scatter_plot(fig, ax[5], lon, lat, truth - input_, cmap="bwr", title=f"{vname} persist err")
    else:
        ax[0].axis("off")
        ax[4].axis("off")
        ax[5].axis("off")


def scatter_plot(fig, ax, lon: np.array, lat: np.array, data: np.array, cmap: str = "viridis", title: Optional[str] = None) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    data : _type_
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    title : _type_, optional
        Title for plot, by default None
    """
    psc = ax.scatter(
        lon,
        lat,
        c=data,
        cmap=cmap,
        s=1,
        alpha=1.0,
        norm=TwoSlopeNorm(vcenter=0.0) if cmap == "bwr" else None,
        rasterized=True,
    )
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((-np.pi / 2, np.pi / 2))

    continents.plot_continents(ax)

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("auto", adjustable=None)
    _hide_axes_ticks(ax)
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
    pc_lon, pc_lat = pc(lon, lat)

    for i in range(nplots):
        ax_ = ax[i] if nplots > 1 else ax
        scatter_plot(fig, ax_, pc_lon, pc_lat, features[..., i])

    return fig
