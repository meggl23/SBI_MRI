
import h5py
import torch
from sbi.utils import BoxUniform

from datetime import datetime

import numpy as np

def save_h5(filename, Params, S, S_raw, snr,parameter_list,compartments):

    if filename is None:
        abbreviations = {
            "DTI": "D",
            "Sticks": "S",
            "Balls": "B",
            "FreeWater": "FW",
        }

        model_name = "".join(
            abbreviations[type(comp).__name__]
            for comp in compartments.values()
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{model_name}_{timestamp}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("Params", data=Params)
        f.create_dataset("S", data=S)
        f.create_dataset("S_raw", data=S_raw)
        f.create_dataset(
            "parameter_names",
            data=parameter_list
        )

        f.attrs["snr"] = -1 if snr is None else snr
        f.attrs["snr_is_none"] = snr is None
    print(f"File was saved as {filename}!")

def read_h5(filename):
    with h5py.File(filename, "r") as f:
        Params = f["Params"][:]
        S = f["S"][:]
        S_raw = f["S_raw"][:]

        parameter_names = [
            x.decode("utf-8")
            for x in f["parameter_names"][:]
        ]

        snr = None if f.attrs["snr_is_none"] else f.attrs["snr"]

    return Params, S, S_raw, parameter_names, snr

def PrepData(P,S,Names):
    print(f'As one of the fractions is redundant we will remove it (1-sum of the rest). We are removing: {Names[0]}')

    Par = torch.tensor(P[:,1:]).float()
    Obs = torch.tensor(S).float()
    Names_cut = Names[1:]

    # Training Ranges
    low = Par.min(axis=0)[0] - 10*torch.sign(Par.min(axis=0)[0])*Par.min(axis=0)[0]
    low = np.clip(low,low,-1)
    high = Par.max(axis=0)[0] + 10*Par.max(axis=0)[0]

    prior_bounds = BoxUniform(low=low, high=high)

    return Par, Obs, Names_cut, prior_bounds

def save_posterior(output, filename=None, compression_level=4,return_dist=False):

    if filename is None:
        timestamp = datetime.now().strftime("Posterior_%Y%m%d_%H%M")
        if( not retur_dist): timestamp = timestamp + ('_mean')
        filename = f"Posterior_{timestamp}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset(
            "posterior",
            data=output,
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,
        )

    print(f"Posterior saved to '{filename}'")


def load_posterior(filename):
    """
    Load a posterior volume from an HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 filename.

    Returns
    -------
    output : np.ndarray
        Posterior array.
    """

    with h5py.File(filename, "r") as f:
        output = f["posterior"][:]

    return output