
from sbi.inference import SNPE
import torch
import tqdm
import numpy as np

import Helpers
from datetime import datetime
import dill as pickle


def Train_Network(Par, Obs,save=True, **kwargs):
    """
    Train a neural posterior estimator using simulation-based inference.

    The supplied parameter and observation pairs are appended to an SNPE
    inference object, which is then trained to estimate the posterior
    distribution. The trained density estimator is converted into a posterior
    object that can subsequently be used for inference.

    Parameters
    ----------
    Par : torch.Tensor
        Simulated model parameters used for training. The first dimension
        corresponds to simulation samples.

    Obs : torch.Tensor
        Simulated observations corresponding to ``Par``.

    **kwargs
        Additional keyword arguments passed directly to ``SNPE.train``.

    Returns
    -------
    NPE_Network : object
        Trained posterior object created by ``SNPE.build_posterior``.
    """
    inference = SNPE()
    inference = inference.append_simulations(Par, Obs)
    density_estimator = inference.train(**kwargs) 
    NPE_Network = inference.build_posterior(density_estimator)

    if save: Save_Network(NPE_Network,Par.shape[0])
    return NPE_Network

def Train_Network_gpu(Par, Obs, batch_size=512,save=True, **kwargs):
    """
    Train a neural posterior estimator using available GPU acceleration.

    CUDA is used when available. If CUDA is unavailable but Apple's MPS
    backend is available, MPS is used instead. If no supported GPU backend is
    found, the function falls back to ``Train_Network`` on the CPU.

    After training, the resulting posterior network is moved back to the CPU
    before being returned.

    Parameters
    ----------
    Par : torch.Tensor
        Simulated model parameters used for training.

    Obs : torch.Tensor
        Simulated observations corresponding to ``Par``.

    batch_size : int, optional
        Training batch size passed to ``SNPE.train`` as
        ``training_batch_size``. Default is 512.

    **kwargs
        Additional keyword arguments passed directly to ``SNPE.train``.

    Returns
    -------
    NPE_Network : object
        Trained posterior object created by ``SNPE.build_posterior`` and moved
        to the CPU before being returned.

    Notes
    -----
    Device selection follows the priority order CUDA, MPS, then CPU.
    """

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        print ("No GPU acceleration found, will resort to CPU version")
        return Train_Network(Par,Obs)

    inference = SNPE(device=device)
    inference = inference.append_simulations(Par, Obs)
    density_estimator = inference.train(training_batch_size = batch_size,**kwargs) 
    NPE_Network = inference.build_posterior(density_estimator)
    NPE_Network.to(device='cpu')

    if save: Save_Network(NPE_Network,Par.shape[0])

    return NPE_Network

def Save_Network(NPE_Network,N):

    def format_n(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:g}M"
        if n >= 1_000:
            return f"{n / 1_000:g}k"
        return str(n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"Network_{format_n(N)}_{timestamp}"

    filename= Helpers.make_data_folder(filename)

    with open(f"{filename}.pickle", "wb") as handle:
        pickle.dump(NPE_Network, handle)

    print(f"Network saved as '{filename}.pickle'")

def Load_Network(filename):

    if "saved_data/" not in filename: filename= Helpers.make_data_folder(filename)

    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(
            f"Network file does not exist: {filename.resolve()}"
        )

    with open(filename, "rb") as handle:
        posterior = pickle.load(handle)

def Infer(Network, Obs, Num_samples=500, batch_size=32, show_tqdm=True):
    """
    Generate posterior samples for a collection of observations.

    Observations are processed in batches and passed to the trained posterior
    network using ``sample_batched``. GPU acceleration is used when available,
    with CUDA preferred over Apple's MPS backend. Posterior samples are moved
    back to the CPU before being returned.

    Parameters
    ----------
    Network : object
        Trained posterior network supporting ``to`` and ``sample_batched``.

    Obs : torch.Tensor
        Observations for which posterior samples should be generated. The
        first dimension corresponds to independent observations.

    Num_samples : int, optional
        Number of posterior samples generated for each observation.
        Default is 500.

    batch_size : int, optional
        Number of observations processed simultaneously during inference.
        Default is 32.

    show_tqdm : bool, optional
        If ``True``, display a progress bar while processing observation
        batches. Default is ``True``.

    Returns
    -------
    p_samples : torch.Tensor
        Posterior samples for all observations. The resulting tensor has shape
        ``(samples, n_observations, n_parameters)``.

    """

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        print ("No GPU acceleration found, will resort to CPU version")
        device = 'cpu'
    p_samples = []
    Obs_mps = Obs.to(device)
    Network.to(device)
    if(show_tqdm):
        for i in tqdm.tqdm(range(0, len(Obs_mps), batch_size),position=0):
            O_batch = Obs_mps[i:i + batch_size]
            s = Network.sample_batched((Num_samples,), O_batch,show_progress_bars=False)
            p_samples.append(s.cpu())
    else:
        for i in range(0, len(Obs_mps),batch_size):
            O_batch = Obs_mps[i:i + batch_size]
            s = Network.sample_batched((Num_samples,), O_batch,show_progress_bars=False)
            p_samples.append(s.cpu())        
    p_samples = torch.cat(p_samples, dim=1)
    Network.to('cpu')

    return p_samples

def Infer_from_volume(Network,Obs,mask = None,batch_size = 32, Num_samples=500, return_dist = False, filename = None,save=False):
    """
    Perform posterior inference on a spatial volume of observations.

    The input volume is optionally restricted using a boolean mask. Valid
    voxels are flattened, posterior inference is performed in batches, and
    the resulting estimates are reshaped back into the original spatial
    dimensions.

    By default, the posterior mean is returned for each voxel. If
    ``return_dist=True``, the full posterior sample distribution is retained.

    Parameters
    ----------
    Network : object
        Trained posterior network used for inference.

    Obs : np.ndarray
        Spatial array of observations. The final dimension contains the
        observation or signal features, while all preceding dimensions define
        the spatial volume.

    mask : np.ndarray of bool or None, optional
        Boolean mask defining which voxels should undergo inference. Its shape
        must match ``Obs.shape[:-1]``. If ``None``, all voxels are included.
        Default is ``None``.

    batch_size : int, optional
        Number of voxels processed simultaneously during posterior inference.
        Default is 32.

    samples : int, optional
        Number of posterior samples generated for each voxel.
        Default is 500.

    return_dist : bool, optional
        If ``False``, return the posterior mean for each voxel. If ``True``,
        retain the complete posterior sample distribution. Default is
        ``False``.

    filename : str or None, optional
        Output filename used when saving the posterior result. If ``None``,
        a filename is generated automatically by the saving function.
        Default is ``None``.

    save : bool, optional
        If ``True``, save the resulting posterior volume to disk.
        Default is ``False``.

    Returns
    -------
    output : np.ndarray
        Spatial posterior result.

        If ``return_dist=False``, the shape is
        ``mask.shape + (n_parameters,)``.

        If ``return_dist=True``, the shape is
        ``mask.shape + (samples, n_parameters)``.

        Voxels outside the mask are filled with ``NaN``.

    Notes
    -----
    When no mask is provided, all spatial locations are included.

    Posterior inference is performed using ``Infer`` with the requested number
    of posterior samples and the specified batch size.
    """

    if mask is None:
        mask = np.ones_like(Obs[...,0]).astype(bool)
    Obs_mask = Obs[mask]                     
    Obs_mask = torch.from_numpy(Obs_mask).float()
    print('Starting inference....')
    samples = Infer(Network,Obs_mask, batch_size=batch_size,Num_samples = Num_samples)
    print('Reshaping result')
    samples_np = samples.numpy()
    N_samples, N_voxels, N_params = samples_np.shape
    if return_dist:
        out_shape = mask.shape + (N_samples, N_params)
        output = np.full(out_shape, np.nan, dtype=np.float32)
        output[mask] = samples_np.transpose(1, 0, 2)
    else:
        mean_np = samples_np.mean(axis=0)  # (N_voxels, N_params)
        out_shape = mask.shape + (N_params,)
        output = np.full(out_shape, np.nan, dtype=np.float32)
        output[mask] = mean_np
        
    print("Saving...")
    if save: Helpers.save_posterior(output, filename=filename,return_dist = return_dist)
    
    return output