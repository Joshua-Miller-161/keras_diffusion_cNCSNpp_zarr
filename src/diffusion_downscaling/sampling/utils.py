"""Some utilities for setting up sampling from pre-set configurations.

Implement a range of schedules and sampling methods for the diffusion process.
Schedules - Karras, exponential, vp, etc.
Sampling - Euler, Heun 2nd order, Heun dpm2.
"""

from .k_sampling import (
    get_sigmas_karras,
    get_sigmas_exponential,
    get_sigmas_polyexponential,
    get_sigmas_vp,
    get_sigmas_ve,
    get_sigmas_karras_sqrt,
)
from .k_sampling import sample_euler, sample_dpm_2, sample_heun
import inspect
import logging

logger = logging.getLogger(__name__)

dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])


def create_sampling_configurations(eval_config, location_config):

    schedule_callable, schedule_config = build_schedule(eval_config.schedule)

    sampling_callable, sampling_config = build_sampling_callable_and_config(
        eval_config.sampler
    )
    location_config_dict = {"location_config": location_config}
    sweep_args_list = combine_configs_and_product_lists(
        schedule_config, sampling_config, location_config_dict
    )

    split_sweep_args = [
        (
            dictfilt(args_dict, schedule_config),
            dictfilt(args_dict, sampling_config),
            dictfilt(args_dict, location_config_dict),
        )
        for args_dict in sweep_args_list
    ]

    return (schedule_callable, sampling_callable), split_sweep_args


def create_inference_sampling_configuration(sampling_config, location_config):
    """Create a single sampling configuration from sampling.sampler params only.

    Unlike create_sampling_configurations (used by sampling_grid_search.py), this
    function uses exclusively sampling.sampler to build both the schedule and sampler
    configs. This ensures inference.py never picks up grid_search list parameters.

    Parameters n and rho are taken from sampler. Schedule-only params (type,
    sigma_min, sigma_max) are also sourced from sampler when present; otherwise
    scalar values from sampling.schedule are used as fallbacks so that existing
    config files (where sampling.schedule is aliased to sampling.grid_search)
    still work correctly.
    """
    sampler_dict = dict(sampling_config.sampler)

    # Extract schedule parameters that live in sampler for inference configs
    n = sampler_dict.pop('n', None)
    rho = sampler_dict.pop('rho', None)

    # For schedule-only params, prefer sampler; fall back to schedule scalar values
    schedule_obj = getattr(sampling_config, 'schedule', None)
    schedule_dict_raw = dict(schedule_obj) if schedule_obj is not None else {}

    schedule_type = sampler_dict.pop(
        'schedule_type', schedule_dict_raw.get('type', 'karras')
    )

    raw_sigma_min = schedule_dict_raw.get('sigma_min', 0.02)
    sigma_min = sampler_dict.pop(
        'sigma_min',
        raw_sigma_min if not isinstance(raw_sigma_min, list) else 0.02,
    )

    raw_sigma_max = schedule_dict_raw.get('sigma_max', 80)
    sigma_max = sampler_dict.pop(
        'sigma_max',
        raw_sigma_max if not isinstance(raw_sigma_max, list) else 80,
    )

    device = schedule_dict_raw.get('device', 'cpu')

    if n is None:
        raw_n = schedule_dict_raw.get('n', 20)
        n = raw_n if not isinstance(raw_n, list) else 20
        logger.warning(
            "n not found in sampling.sampler; falling back to %s. "
            "Add sampler.n to the sampling config for inference.",
            n,
        )

    if rho is None:
        raw_rho = schedule_dict_raw.get('rho', 7.0)
        rho = raw_rho if not isinstance(raw_rho, list) else 7.0
        logger.warning(
            "rho not found in sampling.sampler; falling back to %s. "
            "Add sampler.rho to the sampling config for inference.",
            rho,
        )

    schedule_config = {
        'type': schedule_type,
        'n': n,
        'rho': rho,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'device': device,
    }

    try:
        schedule_callable = _SCHEDULE_LOOKUP[schedule_type]
    except KeyError as e:
        print(f"Schedule {schedule_type} not supported. Is there a typo?")
        raise e

    schedule_config = _filter_callable_kwargs(schedule_callable, schedule_config, "schedule")
    sampling_callable, sampler_config = build_sampling_callable_and_config(sampler_dict)

    location_config_dict = {"location_config": location_config}
    sweep_args_list = combine_configs_and_product_lists(
        schedule_config, sampler_config, location_config_dict
    )

    split_sweep_args = [
        (
            dictfilt(args_dict, schedule_config),
            dictfilt(args_dict, sampler_config),
            dictfilt(args_dict, location_config_dict),
        )
        for args_dict in sweep_args_list
    ]

    return (schedule_callable, sampling_callable), split_sweep_args


_SCHEDULE_LOOKUP = {
    "karras": get_sigmas_karras,
    "karras_sqrt": get_sigmas_karras_sqrt,
    "exponential": get_sigmas_exponential,
    "polyexponential": get_sigmas_polyexponential,
    "vp": get_sigmas_vp,
    "ve": get_sigmas_ve,
}

def _filter_callable_kwargs(callable_obj, config, label):
    signature = inspect.signature(callable_obj)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return config
    allowed_keys = set(signature.parameters.keys())
    filtered = {key: value for key, value in config.items() if key in allowed_keys}
    dropped = sorted(set(config) - allowed_keys)
    if dropped:
        logger.warning(
            "Dropping unsupported %s args for %s: %s",
            label,
            getattr(callable_obj, "__name__", str(callable_obj)),
            ", ".join(dropped),
        )
    return filtered


def build_schedule(schedule_config):
    config = dict(schedule_config)
    schedule_type = config.pop("type")
    try:
        schedule_callable = _SCHEDULE_LOOKUP[schedule_type]
    except KeyError as e:
        print(f"Schedule {schedule_type} not supported. Is there a typo?")
        raise e
    except Exception as e:
        print(f"Schedule {schedule_type} had incorrect config {config}")
        raise e
    config = _filter_callable_kwargs(schedule_callable, config, "schedule")
    return schedule_callable, config


_INTEGRATOR_LOOKUP = {
    'euler': sample_euler,
    'heun': sample_heun,
    'dpm2_heun': sample_dpm_2,
}


def build_sampling_callable_and_config(sampling_config):
    """
    Should return a sampling callable and a
     list of all the requested sampling configurations.
    """
    config = dict(sampling_config)
    schedule_type = config.pop("integrator")
    try:
        sampling_callable = _INTEGRATOR_LOOKUP[schedule_type]
    except KeyError as e:
        print(f"Schedule {schedule_type} not supported. Is there a typo?")
        raise e

    config = _filter_callable_kwargs(sampling_callable, config, "sampler")
    return sampling_callable, config


def collect_sampling_config_lists(sampling_confg):
    scalars = {}
    lists = {}
    for key, value in sampling_confg.items():
        if isinstance(value, list):
            lists[key] = value
        else:
            scalars[key] = value
    return list(dict_itertools_product(lists, scalars))


from itertools import product


# This code allows the user to combine multiple config parameters (specified as lists).
# A product over all possible parameters is taken and fed on to the evaluation utilities.
def combine_configs_and_product_lists(
    schedule_config, sampling_config, location_config
):
    return collect_sampling_config_lists(
        {**schedule_config, **sampling_config, **location_config}
    )


def dict_itertools_product(inp, extra_args):
    return (
        {**dict(zip(inp.keys(), values)), **extra_args}
        for values in product(*inp.values())
    )
