import importlib

import omegaconf

from .models import ContrastiveModel, DiffuserSTDiT, ResNet18, SegDiTTransformer2DModel


def parse_klass_arg(value, full_config):
    """
    Parse an argument value that might represent a class, enum, or basic data type.
    This function tries to dynamically import and resolve nested attributes.
    It also resolves OmegaConf interpolations if found.
    """
    if isinstance(value, str) and "." in value:
        # Check if the value is an interpolation and try to resolve it
        if value.startswith("${") and value.endswith("}"):
            try:
                # Attempt to resolve the interpolation directly using OmegaConf
                value = omegaconf.OmegaConf.resolve(full_config)[value[2:-1]]
            except Exception as e:
                print(f"Error resolving OmegaConf interpolation {value}: {e}")
                return None

        parts = value.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            attr_name = parts[i]
            try:
                module = importlib.import_module(module_name)
                result = module
                for j in range(i, len(parts)):
                    result = getattr(result, parts[j])
                return result
            except ImportError as e:
                continue
            except AttributeError as e:
                print(
                    f"Warning: Could not resolve attribute {parts[j]} from {module_name}, error: {e}"
                )
                continue
        # print(f"Warning: Failed to import or resolve {value}. Falling back to string.")
        return (
            value  # Return the original string if no valid import and resolution occurs
        )
    return value


def instantiate_class_from_config(config, *args, **kwargs):
    """
    Dynamically instantiate a class based on a configuration object.
    Supports passing additional positional and keyword arguments.
    """
    module_name, class_name = config.target.rsplit(".", 1)
    klass = globals().get(class_name)
    # module = importlib.import_module(module_name)
    # klass = getattr(module, class_name)

    # Assuming config might be a part of a larger OmegaConf structure:
    # if not isinstance(config, omegaconf.DictConfig):
    #     config = omegaconf.OmegaConf.create(config)
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    # Resolve args and kwargs from the configuration
    # conf_args = [parse_klass_arg(arg, config) for arg in config.get('args', [])]
    # conf_kwargs = {key: parse_klass_arg(value, config) for key, value in config.get('kwargs', {}).items()}
    conf_kwargs = {
        key: parse_klass_arg(value, config) for key, value in config["args"].items()
    }
    # Combine conf_args with explicitly passed *args
    all_args = list(args)  # + conf_args

    # Combine conf_kwargs with explicitly passed **kwargs
    all_kwargs = {**conf_kwargs, **kwargs}

    # Instantiate the class with the processed arguments
    instance = klass(*all_args, **all_kwargs)
    return instance


def unscale_latents(latents, vae_scaling=None):
    if vae_scaling is not None:
        if latents.ndim == 4:
            v = (1, -1, 1, 1)
        elif latents.ndim == 5:
            v = (1, -1, 1, 1, 1)
        else:
            raise ValueError("Latents should be 4D or 5D")
        latents *= vae_scaling["std"].view(*v)
        latents += vae_scaling["mean"].view(*v)

    return latents

