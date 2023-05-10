import datetime as dt
import os

import numpy as np
import pytorch_lightning as pl
import torch

from aifs.data.era_datamodule import ERA5TestDataModule
from aifs.train.trainer import GraphForecaster
from aifs.train.utils import get_args, setup_exp_logger
from aifs.utils.config import YAMLConfig
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


# def _reshape_predictions(predictions: torch.Tensor, config: YAMLConfig) -> torch.Tensor:
#     """
#     Reshapes the predictions:
#     (batch_size, lat * lon, nvar * plevs, rollout) -> (batch_size, nvar, plevs, lat, lon, rollout)
#     Args:
#         predictions: predictions (already concatenated)
#     Returns:
#         Reshaped predictions (see above)
#     """
#     LOGGER.debug("BEFORE reshape -- predictions.shape = %s", predictions.shape)
#     l = len(config["input:variables:levels"]) if config["input:variables:levels"] is not None else _ERA_PLEV

#     assert predictions.shape[1] == _WB_LAT * _WB_LON, "Predictions tensor doesn't have the expected lat/lon shape!"
#     return rearrange(
#         predictions,
#         "b (h w) (v l) r -> b v l h w r",
#         h=_WB_LAT,
#         w=_WB_LON,
#         v=len(config["input:variables:names"]),
#         l=l,
#         r=config["model:rollout"],
#     )


# def store_predictions(
#     predictions: torch.Tensor,
#     sample_idx: torch.Tensor,
#     config: YAMLConfig,
# ) -> None:
#     plevs = config["input:pl:levels"]
#     plevs.reverse()
#     pl_vars = config["input:pl:names"]
#     for i in range(4):
#         for j in range(13):
#             plevs.append
#     print(sample_idx)
#     for i, idxs in enumerate(sample_idx):
#         # this is loop over batches
#         # need to add loop over batch itself
#         print(idx)
#         if idx % 4 > 0:
#             pass
#         date = dt.datetime(2015, 1, 1) + dt.timedate(hours=6 * idx)
#         filename = os.path.join("/ec/res4/scratch/pamc/era5/o160/zarr/pl", f"test_{date.strftime(format='%Y%m%d%H')}_pl.grib")
#         pl_output = cml.new_grib_output(filename)
#         print(
#             "Writing pressure levels output to %s",
#             os.path.realpath(f"{output}_pl{ext}"),
#         )

#         for step in range(predictions.shape[1]):
#             pl_data = predictions[i, step, :, : 5 * 13].swapaxes(0, 1)
#             metadata = {"type": "fc", "expver": "gnn0", "step": (step + 1) * 6, "date": date}
#             for j, data in enumerate(pl_data):
#                 my_lev = plevs[j % len(plevs)]
#                 my_var = pl_vars[j // len(plevs)]
#                 template_dic = dict(param=my_var, level=my_lev, levtype="pl")
#                 pl_output.write(data, metadata={**metadata, **template_dic})  # template=f,


#     # create a new xarray Dataset with similar coordinates as ds_test, plus the rollout
#     ds_pred_gnn = xr.Dataset()

#     with xr.open_dataset(config["input:variables:prediction:filename"]) as ds_predict:
#         LOGGER.debug(ds_predict)

#         ds_predict = ds_predict.assign_coords({"longitude": (((ds_predict.longitude + 180) % 360) - 180.0)})
#         ds_predict = ds_predict.sortby("longitude").sortby("latitude")

#         for var_idx, varname in enumerate(config["input:variables:names"]):
#             LOGGER.debug(
#                 "varname: %s -- min: %.3f -- max: %.3f",
#                 varname,
#                 predictions[:, var_idx, ...].min(),
#                 predictions[:, var_idx, ...].max(),
#             )
#             plevs: xr.DataArray = (
#                 ds_predict.coords["level"].sel(level=config["input:variables:levels"])
#                 if config["input:variables:levels"] is not None
#                 else ds_predict.coords["level"]
#             )
#             ds_pred_gnn[varname] = xr.DataArray(
#                 data=predictions[:, var_idx, ...].squeeze(dim=1).numpy(),
#                 dims=["time", "level", "latitude", "longitude", "rollout"],
#                 coords=dict(
#                     latitude=ds_predict.coords["latitude"],
#                     longitude=ds_predict.coords["longitude"],
#                     level=plevs,
#                     time=ds_predict.coords["time"].isel(time=sample_idx[0, :]),
#                     rollout=np.arange(0, config["model:rollout"], dtype=np.int32),
#                 ),
#                 attrs=ds_predict[varname].attrs,
#             )
#             ds_pred_gnn[varname] = ds_pred_gnn[varname].sortby("time", ascending=True)

#     comp = dict(zlib=True, complevel=9)
#     encoding = {var: comp for var in ds_pred_gnn.data_vars}
#     ds_pred_gnn.to_netcdf(os.path.join(config["output:basedir"], "predictions/wb_pred_gnn.nc"), encoding=encoding)


# def backtransform_predictions(predictions: torch.Tensor, means: xr.Dataset, sds: xr.Dataset, config: YAMLConfig) -> torch.Tensor:
#     """
#     Transforms the model predictions back into the original data domain.
#     ATM this entails a simple (Gaussian) re-scaling: predictions <- predictions * sds + means.
#     Args:
#         predictions: predictions tensor, shape == (batch_size, lat*lon, nvar*plev, rollout)
#         means, sds: summary statistics calculated from the training dataset
#         config: job configuration
#     Returns:
#         Back-transformed predictions, shape == (batch_size, nvar, plev, lat, lon, rollout)
#     """
#     predictions = _reshape_predictions(predictions, config)
#     LOGGER.debug("AFTER reshape -- predictions.shape = %s", predictions.shape)

#     for ivar, varname in enumerate(config["input:variables:names"]):
#         LOGGER.debug("Varname: %s, Mean: %.3f, Std: %.3f", varname, means[varname].values, sds[varname].values)
#         predictions[:, ivar, ...] = predictions[:, ivar, ...] * sds[varname].values + means[varname].values
#     return predictions


def predict(config: YAMLConfig) -> None:
    """
    Predict entry point.
    Args:
        config: job configuration
        checkpoint_relpath: path to the model checkpoint that you want to restore
                            should be relative to your config["output:basedir"]/config["output:checkpoints:ckpt-dir"]
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    torch.set_float32_matmul_precision("high")

    # create data module (data loaders and data sets)
    dmod = ERA5TestDataModule(config)

    # number of variables (features)
    num_features = len(config["input:pl:names"]) * len(config["input:pl:levels"]) + len(config["input:sfc:names"])
    num_aux_features = config["input:num-aux-features"]
    num_fc_features = num_features - num_aux_features

    loss_scaling = []
    for scl in config["input:loss-scaling-pl"]:
        loss_scaling.extend([scl] * len(config["input:pl:levels"]))
    for scl in config["input:loss-scaling-sfc"]:
        loss_scaling.append(scl)
    assert len(loss_scaling) == num_fc_features
    LOGGER.debug("Loss scaling: %s", loss_scaling)
    loss_scaling = torch.from_numpy(np.array(loss_scaling, dtype=np.float32))

    LOGGER.debug("Total number of prognostic variables: %d", num_fc_features)
    LOGGER.debug("Total number of auxiliary variables: %d", num_aux_features)

    graph_data = torch.load(os.path.join(config["graph:data-basedir"], config["graph:data-file"]))

    model = GraphForecaster(
        graph_data=graph_data,
        metadata=dmod.input_metadata,
        fc_dim=num_fc_features,
        aux_dim=num_aux_features,
        num_levels=len(config["input:pl:levels"]),
        encoder_hidden_channels=config["model:encoder:num-hidden-channels"],
        encoder_out_channels=config["model:encoder:num-out-channels"],
        encoder_num_layers=config["model:encoder:num-layers"],
        encoder_mapper_num_layers=config["model:encoder:mapper-num-layers"],
        mlp_extra_layers=config["model:encoder:mlp-extra-layers"],
        activation=config["model:encoder:activation"],
        rollout=config["model:rollout"],
        save_basedir=os.path.join(
            config["output:basedir"].format(resolution=config["input:resolution"]),
            config["output:plots:plot-dir"],
            timestamp,
        ),
        log_to_wandb=config["model:wandb:enabled"],
        log_to_neptune=config["model:neptune:enabled"],
        log_persistence=False,
        loss_scaling=loss_scaling,
        pl_names=config["input:pl:names"],
    )

    ckpt_path = os.path.join(
        config["output:basedir"].format(resolution=config["input:resolution"]),
        config["output:checkpoints:ckpt-dir"],
        config["model:warm-restart:ckpt-path"],
    )
    LOGGER.debug("Loading checkpoint from %s ...", ckpt_path)

    trainer = pl.Trainer(
        accelerator="gpu" if config["model:num-gpus"] > 0 else "cpu",
        detect_anomaly=config["model:debug:anomaly-detection"],
        strategy=config["model:strategy"],
        devices=config["model:num-gpus"] if config["model:num-gpus"] > 0 else None,
        num_nodes=config["model:num-nodes"],
        precision=config["model:precision"],
        max_epochs=config["model:max-epochs"],
        logger=setup_exp_logger(config),
        log_every_n_steps=config["output:logging:log-interval"],
        limit_predict_batches=config["model:limit-batches:predict"],
        limit_test_batches=config["model:limit-batches:test"],
        use_distributed_sampler=False,
    )

    # run a test loop (calculates & logs test loss and metrics, returns nothing)
    # trainer.test(model, datamodule=dmod, ckpt_path=ckpt_path, verbose=True)

    # run a predict loop on a "predict" dataset (can be the same as "test" or different)
    # this returns the predictions and sample indices
    predict_output = trainer.predict(model, datamodule=dmod, return_predictions=True, ckpt_path=ckpt_path)

    predictions_, sample_idx_ = zip(*predict_output)

    # len(predictions_) = num-predict-batches, predictions_[0].shape = (batch-size, latlon, nvar, rollout)
    # len(sample_idx_) = num-predict-batches, sample_idx_[0].shape = (batch_size, )
    LOGGER.debug("len(predictions_) = %d, predictions_[0].shape = %s", len(predictions_), predictions_[0].shape)
    LOGGER.debug("len(sample_idx_) = %d, sample_idx_[0].shape = %s", len(sample_idx_), sample_idx_[0].shape)

    # predictions.shape = (batch_size * num-predict-batches, latlon, nvar, rollout)
    # sample_idx.shape = (batch_size * num-predict-batches, )
    predictions: torch.Tensor = torch.cat(predictions_, dim=0).float()
    sample_idx = np.concatenate(sample_idx_, axis=-1)

    LOGGER.debug("predictions.shape = %s", predictions.shape)
    LOGGER.debug("sample_idx.shape = %s", sample_idx.shape)
    # for p, i in zip(predictions_, sample_idx_):
    #     store_predictions(p, i, config)

    # *****************
    # TODO:
    # *****************
    # backtransform the predictions to the original data space
    # write the output to disk (zarr / netcdf / ... ?)
    #  - need to keep around the sample_idx info (so we can match the predictions against the ground truth)

    # LOGGER.debug("Backtransforming predictions to the original data space ...")
    # predictions = backtransform_predictions(predictions, dmod.ds_test.mean, dmod.ds_test.sd, config)
    # LOGGER.debug("predictions.shape = %s", predictions.shape)

    # # save data along with observations
    # LOGGER.debug("Storing model predictions to netCDF ...")
    # store_predictions(predictions, sample_idx, config)

    LOGGER.debug("---- DONE. ----")


def main() -> None:
    """Entry point for inference."""
    args = get_args()
    config = YAMLConfig(args.config)
    predict(config)
