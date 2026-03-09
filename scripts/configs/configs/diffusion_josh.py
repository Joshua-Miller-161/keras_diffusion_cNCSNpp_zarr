import ml_collections

from .defaults import get_default_configs

def get_config():
    config = get_default_configs()

    config.run_name = "karras_cNCSNpp_severe_weather_unet_thresh15"
    config.project_name = "karras_cNCSNpp_severe_weather_unet_thresh15"
    config.model_type = "diffusion"
    config.diffusion_type = "karras"

    training = config.training
    training.batch_size = 12
    training.loss_weights = [1.0]

    data = config.data
    data.use_josh_pipeline = True
    data.dataset = "severe_weather_unet"
    data.dataset_name = "severe_weather_unet"
    data.input_transform_dataset = "zarr"
    data.transform_dir = "./transforms/zarr"
    data.filename = "time=1_lat=10_lon=10_threshold=15_train.zarr"
    data.val_filename = "time=1_lat=10_lon=10_threshold=15_val_sub7.zarr"
    data.time_inputs = False
    data.prefetch_factor = 2

    data.predictands = ml_collections.ConfigDict()
    data.predictands.variables = ("precipitation",)
    data.predictands.target_transform_keys = ("sqrturrecen",)

    data.predictors = ml_collections.ConfigDict()
    data.predictors.variables = ["specHum850", "specHum700", "specHum500", "specHum250", "temp850", "temp700", "temp500", "temp250", "vort850", "vort700", "vort500", "vort250", "mslp", "elevation"]
    data.predictors.input_transform_keys = ["stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "stan", "mm"]

    data.target_transform_overrides = ml_collections.ConfigDict()

    # maintain compatibility with existing components that expect a tuple of lists
    data.variables = (list(data.predictors.variables), list(data.predictands.variables))

    return config
