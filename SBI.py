
from sbi.inference import SNPE
import torch
import tqdm

import Helpers


def Train_Network(Par,Obs , **kwargs):
    inference = SNPE()
    inference = inference.append_simulations(Par, Obs)
    density_estimator = inference.train(**kwargs) 
    NPE_Network = inference.build_posterior(density_estimator)
    return NPE_Network

def Train_Network_gpu(Par,Obs,batch_size=512, **kwargs):
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
        

    return NPE_Network

def Infer(Network,Obs,samples=500,batch_size=32):

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
    for i in tqdm.tqdm(range(0, len(Obs_mps), batch_size),position=0):
        O_batch = Obs_mps[i:i + batch_size]
        s = Network.sample_batched((samples,), O_batch,show_progress_bars=False)
        p_samples.append(s.cpu())
    p_samples = torch.cat(p_samples, dim=1)
    Network.to('cpu')

    return p_samples

def InferFromVolume(Network,Obs,mask,batch_size = 32, return_dist = False, filename = None,):
    Obs_mask = Obs[mask]                     
    Obs_mask = torch.from_numpy(Obs_mask).float()
    print('Starting inference....')
    samples = Infer(Network,Obs_mask, batch_size=batch_size)
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
    Helpers.save_posterior(output, filename=filename,return_dist = return_dist)
    
    return output