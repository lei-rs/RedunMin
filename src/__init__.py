from jax import config, devices
config.update("jax_default_device", devices('cpu')[0])
