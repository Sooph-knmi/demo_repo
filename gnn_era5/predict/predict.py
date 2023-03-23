# from typing import List
# import argparse
# import os

# from einops import rearrange
# import numpy as np
# import torch
# import pytorch_lightning as pl
# import xarray as xr

# from gnn_era5.utils.config import YAMLConfig
# from gnn_era5.data.era_datamodule import WeatherBenchTestDataModule
# from gnn_era5.utils.logger import get_logger
# from gnn_era5.train.trainer import GraphForecaster
# from gnn_era5.train.utils import setup_exp_logger, get_args
# from gnn_era5.utils.constants import _ERA_PLEV, _WB_LAT, _WB_LON


# LOGGER = get_logger(__name__)


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
#     """
#     Stores the model predictions into a netCDF file.
#     Args:
#         predictions: predictions tensor, shape == (batch_size, nvar, plev, lat, lon, rollout)
#         sample_idx: sample indices (used to match samples to valid time points in the reference predict dataset)
#         config: job configuration
#     TODO: add support for parallel writes with Dask (?)
#     """
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


# def predict(config: YAMLConfig, checkpoint_relpath: str) -> None:
#     """
#     Predict entry point.
#     Args:
#         config: job configuration
#         checkpoint_relpath: path to the model checkpoint that you want to restore
#                             should be relative to your config["output:basedir"]/config["output:checkpoints:ckpt-dir"]
#     """
#     # create data module (data loaders and data sets)
#     dmod = WeatherBenchTestDataModule(config)

#     # number of variables (features)
#     num_features = dmod.ds_test.nlev * dmod.ds_test.nvar
#     LOGGER.debug("Number of variables: %d", num_features)
#     LOGGER.debug("Number of auxiliary (time-independent) variables: %d", dmod.const_data.nconst)

#     # learning rate multiplier when running single-node, multi-GPU and/or multi-node
#     total_gpu_count = config["model:num-nodes"] * config["model:num-gpus"]
#     LOGGER.debug("Total GPU count: %d - NB: the learning rate will be scaled by this factor!")
#     LOGGER.debug("Effective learning rate: %.3e", total_gpu_count * config["model:learn-rate"])

#     model = GraphForecaster(
#         era5_lat_lons=dmod.const_data.era5_latlons,
#         h3_lat_lons=dmod.const_data.h3_latlons,
#         fc_dim=num_features,
#         aux_dim=dmod.const_data.nconst,
#         encoder_num_layers=config["model:encoder:num-layers"],
#         encoder_num_heads=config["model:encoder:num-heads"],
#         encoder_activation=config["model:encoder:activation"],
#         use_dynamic_context=True,
#         lr=total_gpu_count * config["model:learn-rate"],
#         rollout=config["model:rollout"],
#     )

#     # TODO: restore model from checkpoint
#     checkpoint_filepath = os.path.join(config["output:basedir"], config["output:checkpoints:ckpt-dir"], checkpoint_relpath)
#     model = GraphForecaster.load_from_checkpoint(checkpoint_filepath)

#     trainer = pl.Trainer(
#         accelerator="gpu" if config["model:num-gpus"] > 0 else "cpu",
#         detect_anomaly=config["model:debug:anomaly-detection"],
#         strategy=config["model:strategy"],
#         devices=config["model:num-gpus"] if config["model:num-gpus"] > 0 else None,
#         num_nodes=config["model:num-nodes"],
#         precision=config["model:precision"],
#         max_epochs=config["model:max-epochs"],
#         logger=setup_exp_logger(config),
#         log_every_n_steps=config["output:logging:log-interval"],
#         limit_test_batches=config["model:limit-batches:test"],
#         limit_predict_batches=config["model:limit-batches:predict"],
#         # we have our own DDP-compliant sampler logic baked into the dataset
#         replace_sampler_ddp=False,
#     )

#     # run a test loop (calculates test_wmse, returns nothing)
#     trainer.test(model, datamodule=dmod)

#     # run a predict loop on a different dataset - this doesn't calculate the WMSE but returns the predictions and sample indices
#     predict_output = trainer.predict(model, datamodule=dmod, return_predictions=True)

#     predictions_, sample_idx_ = zip(*predict_output)
#     LOGGER.debug("len(predictions_) = %d, predictions_[0].shape = %s", len(predictions_), predictions_[0].shape)
#     LOGGER.debug("len(sample_idx_) = %d, sample_idx_[0].shape = %s", len(sample_idx_), sample_idx_[0].shape)

#     predictions: torch.Tensor = torch.cat(predictions_, dim=0).float()
#     sample_idx = np.concatenate(sample_idx_, axis=-1)  # shape == (rollout + 1, num_pred_samples)
#     LOGGER.debug("predictions.shape = %s", predictions.shape)
#     LOGGER.debug("sample_idx.shape = %s", sample_idx.shape)

#     LOGGER.debug("Backtransforming predictions to the original data space ...")
#     predictions = backtransform_predictions(predictions, dmod.ds_test.mean, dmod.ds_test.sd, config)
#     LOGGER.debug("predictions.shape = %s", predictions.shape)

#     # save data along with observations
#     LOGGER.debug("Storing model predictions to netCDF ...")
#     store_predictions(predictions, sample_idx, config)

#     LOGGER.debug("---- DONE. ----")


# def get_args() -> argparse.Namespace:
#     """Returns a namespace containing the command line arguments"""
#     parser = argparse.ArgumentParser()
#     required_args = parser.add_argument_group("required arguments")
#     required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
#     required_args.add_argument(
#         "--checkpoint", required=True, help="Path to the model checkpoint file (located under output-basedir/chkpt-dir)."
#     )
#     return parser.parse_args()


# def main() -> None:
#     """Entry point for inference."""
#     args = get_args()
#     config = YAMLConfig(args.config)
#     predict(config, args.checkpoint)
