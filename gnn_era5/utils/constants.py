# An assortment of constants

# # fixed shapes of the xarray WB data
_ERA_PLEV = 13
_ERA_O160_LATLON = 108160

# ... other stuff
_DL_PREFETCH_FACTOR = 2

# netCDF compression level
_NC_COMPRESS_LEVEL = 9  # max

# blh, msl, z, lsm
_NORMALIZERS_2D = ["max", "std", "max", "none"]

# plotting
_IDXVARS_TO_PLOT = [2, 15, 28, 41, 54, 67, 80, 81]
_NUM_VARS_TO_PLOT = len(_IDXVARS_TO_PLOT)
_NAM_VARS_TO_PLOT = ["q850", "t850", "u850", "v850", "w850", "z850", "msl", "blh"]
_NUM_PLOTS_PER_SAMPLE = 6
