import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from aifs.utils.constants import _IDXVARS_TO_PLOT, _NAM_VARS_TO_PLOT, _NUM_PLOTS_PER_SAMPLE, _NUM_VARS_TO_PLOT
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


def init_plot_settings():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 10

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title


def _hide_axes_ticks(ax, x_axis: bool = True, y_axis: bool = True) -> None:
    # hide x/y-axis ticks
    if x_axis:
        plt.setp(ax.get_xticklabels(), visible=False)
    if y_axis:
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

    scatter_plot(fig, ax[0], pc, lat, lon, input_, title=f"{vname} input")
    scatter_plot(fig, ax[1], pc, lat, lon, truth, title=f"{vname} target")
    scatter_plot(fig, ax[2], pc, lat, lon, pred, title=f"{vname} pred")
    scatter_plot(fig, ax[3], pc, lat, lon, truth - pred, cmap="bwr", title=f"{vname} pred err")
    scatter_plot(fig, ax[4], pc, lat, lon, pred - input_, cmap="bwr", title=f"{vname} increment [pred - input]")
    scatter_plot(fig, ax[5], pc, lat, lon, truth - input_, cmap="bwr", title=f"{vname} persist err")


def scatter_plot(fig, ax, pc, lat, lon, x, cmap="viridis", title=None) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids."""
    ax.set_global()
    ax.add_feature(cf.COASTLINE, edgecolor="black", linewidth=0.5)

    psc = ax.scatter(
        x=lon,
        y=lat,
        c=x,
        cmap=cmap,
        s=1,
        alpha=1.0,
        transform=pc,
        norm=TwoSlopeNorm(vcenter=0.0) if cmap == "bwr" else None,
    )
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("auto", adjustable=None)
    fig.colorbar(psc, ax=ax)


def plot_rank_histograms(rh: np.ndarray) -> Figure:
    """Plots one rank histogram per target variable"""
    fig, ax = plt.subplots(1, _NUM_VARS_TO_PLOT, figsize=(_NUM_VARS_TO_PLOT * 4.5, 4))
    n_ens = rh.shape[0] - 1
    rh = rh.astype(float)

    for vix, idx in enumerate(_IDXVARS_TO_PLOT):
        vname = _NAM_VARS_TO_PLOT[vix]
        rh_ = rh[:, idx]
        ax[vix].bar(np.arange(0, n_ens + 1), rh_ / rh_.sum(), linewidth=1, color="blue", width=0.7)
        ax[vix].hlines(rh_.mean() / rh_.sum(), xmin=-0.5, xmax=n_ens + 0.5, linestyles="--", colors="red")
        ax[vix].set_title(f"{vname} ranks")
        ax[vix].set_xlabel("Ensemble member index")
        _hide_axes_ticks(ax[vix], x_axis=False)

    return fig


def plot_predicted_ensemble(
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
    n_plots_x, n_plots_y = _NUM_VARS_TO_PLOT, nens + 4  # we also plot the truth, ensemble mean, mean error and spread
    LOGGER.debug("n_plots_x = %d, n_plots_y = %d", n_plots_x, n_plots_y)

    figsize = (n_plots_y * 4, n_plots_x * 3)
    fig, ax = plt.subplots(n_plots_x, n_plots_y, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    pc = ccrs.PlateCarree()

    for vix, idx in enumerate(_IDXVARS_TO_PLOT):
        vname = _NAM_VARS_TO_PLOT[vix]
        yt = y_true[..., idx].squeeze()
        yp = y_pred[..., idx].squeeze()
        ax_ = ax[vix, :] if n_plots_x > 1 else ax
        plot_ensemble(fig, ax_, pc, latlons, yt, yp, vname)

    return fig


def plot_kcrps(latlons: np.ndarray, pkcrps: np.ndarray) -> Figure:
    """
    Plots pointwise KCRPS values
    latlons: lat/lon coordinates array, shape (lat*lon, 2)
    pkcrps: array of pointwise kcrps values, shape (nvar, latlon)
    """
    LOGGER.debug("latlons.shape = %s, pkcrps.shape = %s", latlons.shape, pkcrps.shape)
    assert latlons.shape[0] == pkcrps.shape[1], "OOPS - shape mismatch!"

    fig, ax = plt.subplots(1, _NUM_VARS_TO_PLOT, figsize=(_NUM_VARS_TO_PLOT * 4, 3), subplot_kw={"projection": ccrs.PlateCarree()})
    lat, lon = latlons[:, 0], latlons[:, 1]
    for vix, idx in enumerate(_IDXVARS_TO_PLOT):
        vname = _NAM_VARS_TO_PLOT[vix]
        pkcrps_ = pkcrps[idx, :].squeeze()
        scatter_plot(fig, ax[vix], ccrs.PlateCarree(), lat, lon, pkcrps_, title=f"{vname} kCRPS")
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
    """
    Use this when plotting ensembles, where each member is defined on "flat" (reduced Gaussian) grids.
    """

    lat, lon = latlons[:, 0], latlons[:, 1]
    nens = pred.shape[ens_dim]
    ens_mean, ens_sd = pred.mean(axis=ens_dim), pred.std(axis=ens_dim)

    # ensemble mean
    scatter_plot(fig, ax[0], pc, lat, lon, truth, title=f"{vname} target")
    # ensemble mean
    scatter_plot(fig, ax[1], pc, lat, lon, ens_mean, title=f"{vname} pred mean")
    # ensemble spread
    scatter_plot(fig, ax[2], pc, lat, lon, ens_mean - truth, cmap="bwr", title=f"{vname} mean err")
    # ensemble mean error
    scatter_plot(fig, ax[3], pc, lat, lon, ens_sd, title=f"{vname} pred sd")
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
