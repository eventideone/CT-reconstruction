import numpy as np
import logging
import os
import traceback
from utils.config import parse_args_and_config
from runners.diffusion import Diffusion


def my_recon(file):
    data = np.load(file)
    slices, vd = to_slice(data)

    # process

    if Diffusion.instance is None:
        args, config = parse_args_and_config()
        if file.find('low') != -1:
            args.sigma_0 = 0.05
        runner = Diffusion(args, config)
    else:
        runner = Diffusion.instance

    logging.info("Now file = {}".format(file))

    try:
        volume = runner.sample(slices)

    except Exception:
        logging.error(traceback.format_exc())

    volume = to_volume(volume, vd)

    return volume


def to_slice(volume):
    vd = np.zeros([256, 2], np.float32)
    slices = np.zeros([256, 256, 256], np.uint8)
    for i in range(256):
        slice = volume[i, :, :]
        vmax = np.max(slice)
        vmin = np.min(slice)
        slice = (slice - vmin) / (vmax - vmin)
        slices[i, :, :] = (slice * 255).astype(np.uint8)
        vd[i, :] = vmax, vmin
    return slices, vd


def to_volume(slices, vd):
    volume = np.zeros((256, 256, 256), dtype=np.float32)
    for i in range(256):
        A = slices[i, :, :].astype(np.float32)  # / 255
        vmax = vd[i, 0]
        vmin = vd[i, 1]
        A = A * (vmax - vmin) + vmin
        volume[i, :, :] = A
    return volume


def mse(a, b):
    MSE = np.mean((a - b) ** 2)
    logging.info(MSE)
    return MSE
