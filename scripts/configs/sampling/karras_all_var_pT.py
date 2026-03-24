import ml_collections

def get_sampling_config():

    config = ml_collections.ConfigDict()
    config.batch_size = 10
    # config.output_format = "n_20_churn_10_snoise_1.005_rho_3.5_fixed"
    config.output_format = "n_{n}_churn_{s_churn}_snoise_{s_noise}_smin{s_tmin}_rho_{rho}"
    config.eval_indices = {
        "split": [0, 1],
    }

    config.eval = ml_collections.ConfigDict()
    eval = config.eval
    eval.eval_output_dir = "diff"
    config.eval_dataset = "/home/data/unet_splits/test_split7_szn12.zarr"

    eval.checkpoint_name = "/home/temp/checkpoints/unet_splits/PM_split7/epoch=25-val_loss=0.0195.ckpt" # diffusion_

    eval.n_samples = 6
    # eval.location_config = 'colorado'

    config.sampling = ml_collections.ConfigDict()
    sampling = config.sampling

    sampling.sampler = ml_collections.ConfigDict()
    sampler = sampling.sampler
    sampler.integrator = "dpm2_heun"
    sampler.n = 20
    sampler.rho = 7
    sampler.s_churn = 10
    sampler.s_noise = 1.005
    sampler.s_tmin = 0.04
    sampler.s_tmax = 50

    sampling.grid_search = ml_collections.ConfigDict()
    grid_search = sampling.grid_search
    grid_search.type = "karras"
    grid_search.s_churn = [6, 8, 10, 12, 14]
    grid_search.s_noise = [0.985, 0.995, 1.005, 1.015, 1.025]
    grid_search.n = [12, 16, 20, 24, 28]
    grid_search.rho = [3, 5, 7, 9, 11]
    grid_search.sigma_min = 0.02
    grid_search.sigma_max = 80

    # Alias for compatibility with existing sampling utilities.
    sampling.schedule = grid_search

    return config
