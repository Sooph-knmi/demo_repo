from typing import Dict
from typing import Optional
from typing import Tuple

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
    SMALL_SIZE = 10
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

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        xt = x[..., variable_idx].squeeze()
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

    scatter_plot(fig, ax[0], lon, lat, input_, title=f"{vname} input")
    scatter_plot(fig, ax[1], lon, lat, truth, title=f"{vname} target")
    scatter_plot(fig, ax[2], lon, lat, pred, title=f"{vname} pred")
    scatter_plot(fig, ax[3], lon, lat, truth - pred, cmap="bwr", title=f"{vname} pred err")
    scatter_plot(fig, ax[4], lon, lat, pred - input_, cmap="bwr", title=f"{vname} increment [pred - input]")
    scatter_plot(fig, ax[5], lon, lat, truth - input_, cmap="bwr", title=f"{vname} persist err")


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


def plot_rank_histograms(
    parameters: Dict[int, str],
    rh: np.ndarray,
) -> Figure:
    """Plots one rank histogram per target variable."""
    fig, ax = plt.subplots(1, len(parameters), figsize=(len(parameters) * 4.5, 4))
    n_ens = rh.shape[0] - 1
    rh = rh.astype(float)

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        rh_ = rh[:, variable_idx]
        ax[plot_idx].bar(np.arange(0, n_ens + 1), rh_ / rh_.sum(), linewidth=1, color="blue", width=0.7)
        ax[plot_idx].hlines(rh_.mean() / rh_.sum(), xmin=-0.5, xmax=n_ens + 0.5, linestyles="--", colors="red")
        ax[plot_idx].set_title(f"{variable_name} ranks")
        # ax[plot_idx].set_xlabel("Ens member index")
        _hide_axes_ticks(ax[plot_idx])

    return fig


def plot_predicted_ensemble(
    parameters: Dict[int, str],
    latlons: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure:
    """Plots data for one ensemble sample.

    Args:
        latlons: lat/lon coordinates array, shape (lat*lon, 2)
        x, y_true, y_pred: arrays of shape (nens, latlon, nvar)
    Returns:
        The figure object handle.
    """
    nens = y_pred.shape[0]
    n_plots_x, n_plots_y = len(parameters), nens + 4  # we also plot the truth, ensemble mean, mean error and spread
    LOGGER.debug("n_plots_x = %d, n_plots_y = %d", n_plots_x, n_plots_y)

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)
    pc = EquirectangularProjection()

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        yt = y_true[..., variable_idx].squeeze()
        yp = y_pred[..., variable_idx].squeeze()
        ax_ = ax[plot_idx, :] if n_plots_x > 1 else ax
        plot_ensemble(fig, ax_, pc, latlons, yt, yp, variable_name)

    return fig


def plot_ensemble_initial_conditions(
    parameters: Dict[int, str],
    latlons: np.ndarray,
    y_pert: np.ndarray,
) -> Figure:
    """Plots data for one ensemble sample.

    Args:
        latlons: lat/lon coordinates array, shape (lat*lon, 2)
        x, y_true, y_pred: arrays of shape (nens, latlon, nvar)
    Returns:
        The figure object handle.
    """
    nens = y_pert.shape[0]
    n_plots_x, n_plots_y = len(parameters), nens + 1  # plot the mean and perturbations

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize)
    pc = EquirectangularProjection()

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        yp = y_pert[..., variable_idx].squeeze()
        LOGGER.debug("Variable idx %d name %s -- yp.shape = %s", variable_idx, variable_name, yp.shape)
        ax_ = ax[plot_idx, :] if n_plots_x > 1 else ax
        plot_ensemble_ic(fig, ax_, pc, latlons, yp, variable_name)

    return fig


def plot_kcrps(parameters: Dict[str, int], latlons: np.ndarray, pkcrps: np.ndarray) -> Figure:
    """
    Plots pointwise KCRPS values
    latlons: lat/lon coordinates array, shape (lat*lon, 2)
    pkcrps: array of pointwise kcrps values, shape (nvar, latlon)
    """
    assert latlons.shape[0] == pkcrps.shape[1], "Error: shape mismatch!"

    fig, ax = plt.subplots(1, len(parameters), figsize=(len(parameters) * 4, 3))
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc = EquirectangularProjection()

    for plot_idx, (variable_idx, variable_name) in enumerate(parameters.items()):
        pkcrps_ = pkcrps[variable_idx, :].squeeze()
        ax_ = ax[plot_idx] if len(parameters) > 1 else ax
        scatter_plot(fig, ax_, pc, lat, lon, pkcrps_, title=f"{variable_name} kCRPS")
    return fig


def plot_ensemble(
    fig,
    ax,
    pc,
    latlons: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    vname: str,
    ens_dim: int = 0,
) -> None:
    """Use this when plotting ensembles, where each member is defined on "flat" (reduced
    Gaussian) grids."""

    lat, lon = latlons[:, 0], latlons[:, 1]
    nens = pred.shape[ens_dim]
    ens_mean, ens_sd = pred.mean(axis=ens_dim), pred.std(axis=ens_dim)

    LOGGER.debug("latlons.shape = %s truth.shape = %s pred.shape = %s", latlons.shape, truth.shape, pred.shape)

    # ensemble mean
    scatter_plot(fig, ax[0], pc, lat, lon, truth, title=f"{vname} target")
    # ensemble mean
    scatter_plot(fig, ax[1], pc, lat, lon, ens_mean, title=f"{vname} pred mean")
    # ensemble spread
    scatter_plot(fig, ax[2], pc, lat, lon, ens_mean - truth, cmap="bwr", title=f"{vname} ens mean err")
    # ensemble mean error
    scatter_plot(fig, ax[3], pc, lat, lon, ens_sd, title=f"{vname} ens sd")
    # ensemble members (difference from mean)
    for i_ens in range(nens):
        scatter_plot(
            fig,
            ax[i_ens + 4],
            pc,
            lat,
            lon,
            np.take(pred, i_ens, axis=ens_dim) - ens_mean,
            cmap="bwr",
            title=f"{vname}_{i_ens + 1} - mean",
        )


def plot_ensemble_ic(
    fig,
    ax,
    pc,
    latlons: np.ndarray,
    pert: np.ndarray,
    vname: str,
    ens_dim: int = 0,
) -> None:
    lat, lon = latlons[:, 0], latlons[:, 1]
    nens = pert.shape[ens_dim]

    ens_ic_mean = pert.mean(axis=ens_dim)
    scatter_plot(fig, ax[0], pc, lat, lon, ens_ic_mean.squeeze(), cmap="viridis", title=f"{vname}_mean")

    # ensemble ICs
    for i_ens in range(nens):
        scatter_plot(
            fig,
            ax[i_ens + 1],
            pc,
            lat,
            lon,
            np.take(pert, i_ens, axis=ens_dim) - ens_ic_mean,
            cmap="bwr",
            title=f"{vname}_{i_ens + 1}",
        )


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


def plot_spread_skill(
    parameters: Dict[int, str],
    ss_metric: Tuple[np.ndarray, np.ndarray],
    time_step: int,
) -> Figure:
    nplots = len(parameters)
    figsize = (nplots * 5, 4)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    assert isinstance(ss_metric, tuple), f"Expected a tuple and got {type(ss_metric)}!"
    assert len(ss_metric) == 2, f"Expected a 2-tuple and got a {len(ss_metric)}-tuple!"
    assert (
        ss_metric[0].shape[1] == nplots
    ), f"Shape mismatch in the RMSE metric: expected (..., {nplots}) and got {ss_metric[0].shape}!"
    assert (
        ss_metric[0].shape == ss_metric[1].shape
    ), f"RMSE and spread metric shapes do not match! {ss_metric[0].shape} and {ss_metric[1].shape}"

    rmse, spread = ss_metric[0], ss_metric[1]
    rollout = rmse.shape[0]
    x = np.arange(1, rollout + 1) * time_step

    for i, (_, pname) in enumerate(parameters.items()):
        ax_ = ax[i] if nplots > 1 else ax
        ax_.plot(x, rmse[:, i], "-o", color="red", label="mean RMSE")
        ax_.plot(x, spread[:, i], "-o", color="blue", label="spread")
        ax_.legend()
        ax_.set_title(f"{pname} spread-skill")
        ax_.set_xticks(x)
        ax_.set_xlabel("Lead time [hrs]")

    return fig


def plot_spread_skill_bins(
    parameters: Dict[int, str],
    ss_metric: Tuple[np.ndarray, np.ndarray],
    time_step: int,
) -> Figure:
    nplots = len(parameters)
    figsize = (nplots * 5, 4)
    fig, ax = plt.subplots(1, nplots, figsize=figsize)

    assert isinstance(ss_metric, tuple), f"Expected a tuple and got {type(ss_metric)}!"
    assert len(ss_metric) == 2, f"Expected a 2-tuple and got a {len(ss_metric)}-tuple!"
    assert (
        ss_metric[0].shape[1] == nplots
    ), f"Shape mismatch in the RMSE metric: expected (..., {nplots}) and got {ss_metric[0].shape}!"
    assert (
        ss_metric[0].shape == ss_metric[1].shape
    ), f"RMSE and spread metric shapes do not match! {ss_metric[0].shape} and {ss_metric[1].shape}"

    bins_rmse, bins_spread = ss_metric[0], ss_metric[1]
    rollout = bins_rmse.shape[0]

    for i, (_, pname) in enumerate(parameters.items()):
        ax_ = ax[i] if nplots > 1 else ax
        for j in range(rollout):
            (line,) = ax_.plot(bins_spread[j, i, :], bins_rmse[j, i, :], "-", label=str((j + 1) * time_step) + " hr")
            ax_.plot(bins_spread[j, i, :], bins_spread[j, i, :], "--", color="black", label="__nolabel__")
            ax_.set_aspect("equal")
        ax_.legend()
        ax_.set_title(f"{pname} spread-skill binned")
        ax_.set_xlabel("Spread")
        ax_.set_ylabel("Skill")

    return fig
