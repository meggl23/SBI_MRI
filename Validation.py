import SBI
import Helpers

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tqdm

def ParameterEval_TS(Model, validation_snr, k_fold=5, max_samples=100_000, test_samples=10_000, posterior_samples=500, fidelity=5,maximum_training_epochs = 100):
    if max_samples < 2000:
        raise ValueError(f"Max samples is not enough. Received {max_samples}; at least 2000 are required.")

    if k_fold < 1:
        raise ValueError("k_fold must be at least 1")

    if fidelity < 2:
        raise ValueError("fidelity must be at least 2")

    print("~" * 20)
    print("We will perform a study on the number of training samples we might need")

    N = np.unique(np.geomspace(1000, max_samples, fidelity).astype(int))

    # Generate one shared test set
    test_rng = np.random.default_rng(2026)

    Params_test, Signals_test = Model.sample_and_simulation(
        test_samples, parallel=True, Save=False, rng=test_rng, custom_snr=validation_snr)

    Par_test, Obs_test, _, _ = Helpers.PrepData(Params_test, Signals_test, Model.parameter_list)

    kFoldMetrics = []

    for fold in tqdm.tqdm(range(k_fold), position=0, leave=True):
        train_rng = np.random.default_rng(42 + fold)

        Params_train, Signals_train = Model.sample_and_simulation(
            max_samples, parallel=True, Save=False, rng=train_rng, custom_snr=validation_snr)

        Par_train, Obs_train, Names, _ = Helpers.PrepData(
            Params_train, Signals_train, Model.parameter_list)

        AllMetrics = {}

        for n_samples in N:
            network = SBI.Train_Network_gpu(
                Par_train[:n_samples], Obs_train[:n_samples],
                max_num_epochs=maximum_training_epochs, stop_after_epochs=maximum_training_epochs)

            GuessParams = SBI.Infer(
                network, Obs_test, samples=posterior_samples)

            Metrics = Performance(
                Par_test, GuessParams, Model)

            for key, value in Metrics.items():
                AllMetrics.setdefault(key, []).append(value)

            del network
            del GuessParams

        kFoldMetrics.append(AllMetrics)

    PlotPerformance(kFoldMetrics, N, Names)

    return kFoldMetrics, N

def ParameterEval_PS(Model, validation_snr, k_fold=5, max_samples=1000, training_samples=10_000, test_samples=10_000, fidelity=5):

    if max_samples < 10:
        raise ValueError(f"max_samples must be at least 10, but received {max_samples}")

    print("~" * 20)
    print("We will perform a study on the number of posterior samples we might need")

    # Generate one shared test set
    test_rng = np.random.default_rng(2026)
    Params_test, Signals_test = Model.sample_and_simulation(test_samples, parallel=True, Save=False, rng=test_rng, custom_snr=validation_snr)
    Par_test, Obs_test, _, _ = Helpers.PrepData(Params_test, Signals_test, Model.parameter_list)

    # Generate the training set and train one network
    train_rng = np.random.default_rng(45)
    Params_train, Signals_train = Model.sample_and_simulation(training_samples, parallel=True, Save=False, rng=train_rng, custom_snr=validation_snr)
    Par_train, Obs_train, Names, _ = Helpers.PrepData(Params_train, Signals_train, Model.parameter_list)

    Network = SBI.Train_Network_gpu(Par_train, Obs_train)

    N = np.unique(np.geomspace(10, max_samples, fidelity).astype(int))
    kFoldMetrics = []

    for _ in range(k_fold):
        AllMetrics = {}

        for posterior_samples in N:
            GuessParams = SBI.Infer(Network, Obs_test, samples=posterior_samples)
            Metrics = Performance(Par_test, GuessParams, Model)

            for key, value in Metrics.items():
                AllMetrics.setdefault(key, []).append(value)

            del GuessParams

        kFoldMetrics.append(AllMetrics)

    PlotPerformance(kFoldMetrics, N,Names)

    return kFoldMetrics, N

def PlotPerformance(Metrics,N,Names):
    showlegend = False
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*", "h", "H", "p", "8", "d"]

    fig,axs = plt.subplots(2,3,figsize=(32,16))
    ax = axs.ravel()

    for a,k in zip(ax,Metrics[0].keys()):
        X = np.stack([np.array(M[k]) for M in Metrics])
        mean = X.mean(axis=0)

        sem = sp.stats.sem(X,axis=0)
        if(mean.ndim == 2): 
            for y, yerr in zip(mean.T, sem.T):
                a.errorbar(N, y, yerr=yerr,lw=3,alpha=0.5,c='gray')
        else:
            a.errorbar(N, mean, yerr=sem,lw=3,alpha=0.5,c='gray')

        

        if(k=='Coverage'):
            a.fill_between([N[0],N[-1]],[0.94,0.94],[0.96,0.96],alpha=0.5,color='gray')
            mask = (mean < 0.96) & (mean >0.94)

            converged = np.all(mask,axis=tuple(range(1, mean.ndim)))
            idx = np.flatnonzero(converged)
            idx = idx[0] if idx.size else None
        else:
            delta = np.log(mean[1:]) - np.log(mean[:-1])
            delta_N = np.log(N[1:]) - np.log(N[:-1])
            
            derror_dN = np.concatenate(
                [np.ones_like(mean[:1]), (delta.T / delta_N).T],
                axis=0
            )
            
                
            mask = np.abs(derror_dN) < 0.05
        
        converged = np.all(mask,axis=tuple(range(1, derror_dN.ndim)))
        true_idx = np.flatnonzero(converged)
        idx = next(
            (i for i in true_idx if np.all(converged[i:])),
            None
        )

        N_shaped = np.asarray(N).reshape(
            (len(N),) + (1,) * (mean.ndim - 1)
        )
        
        N_grid = np.broadcast_to(N_shaped, mean.shape)
        if mean.ndim == 1:
            a.plot(
                N_grid[mask],
                mean[mask],
                c="green",
                marker="o",
                ls="None",
            )
            a.plot(
                N_grid[~mask],
                mean[~mask],
                c="red",
                marker="o",
                ls="None",
            )

        else:
            parameter_colors = plt.cm.tab20(np.linspace(0, 1, mean.shape[1]))
            markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*"]
            for i in range(mean.shape[1]):
                marker = markers[i % len(markers)]
                a.plot(
                    N[mask[:, i]],
                    mean[mask[:, i], i],
                    c="green",
                    marker=marker,
                    ls="None",
                    label=Names[i],
                )
                a.plot(
                    N[~mask[:, i]],
                    mean[~mask[:, i], i],
                    c="red",
                    marker=marker,
                    ls="None",
                )
            if(not showlegend):
                a.legend()
                showlegend = True
    
        a.semilogx()

        
        if idx is not None:
            a.set_title(k+ f' (Point of convergence: {N[idx]})')
        else:
            a.set_title(k+ f' (Not converged/good coverage)')


def Performance(TrueParams,PosteriorSamples,Model,Mode_eval = False):
    TrueParams = TrueParams.numpy()
    N = PosteriorSamples.shape[1]
    NumSamples = PosteriorSamples.shape[0]
    #Point error - evaluated with mean
    Err_mean = (abs(PosteriorSamples.mean(axis=0)-TrueParams)/(TrueParams.max(axis=0)-TrueParams.min(axis=0))).mean(axis=0)
    #Point error - evaluated with histo
    lower, upper = np.quantile(PosteriorSamples, [0.025, 0.975], axis=0)
    Coverage = np.sum((lower < TrueParams) &  (TrueParams < upper),axis=0) / N
    #Width
    Width = ((upper - lower)/(TrueParams.max(axis=0)-TrueParams.min(axis=0))).mean(axis=0)
    #Signal reconstruction with mean 
    Reconstruct_Paramters = PosteriorSamples.mean(axis=0).numpy()
    Reconstruct_Paramters = np.concatenate([
        (1 - np.sum(Reconstruct_Paramters[:, :(len(Model.compartments) - 1)], axis=1))[:, None],
        Reconstruct_Paramters
    ], axis=1)

    Reconstruct_TrueParamters = np.concatenate([
        (1 - np.sum(TrueParams[:, :(len(Model.compartments) - 1)], axis=1))[:, None],
        TrueParams
    ], axis=1)

    GuessSignals = Model.simulation(Reconstruct_Paramters)

    TrueSignals  = Model.simulation(Reconstruct_TrueParamters)
    SignalRMSE = np.sqrt(((TrueSignals - GuessSignals)**2).mean(axis=1)).mean()
    T = TrueSignals - TrueSignals.mean(axis=1, keepdims=True)
    G = GuessSignals - GuessSignals.mean(axis=1, keepdims=True)
    SignalCorr = (np.sum(T * G, axis=1) / np.sqrt(np.sum(T**2, axis=1) * np.sum(G**2, axis=1))).mean()
    if(Mode_eval):
        if NumSamples < 50:
            raise ValueError(
                f"Number of samples must exceed 50 (Currently {NumSamples})."
            )    
        Histo_est = np.array([[histogram_mode(PosteriorSamples[:, j, i])  
                          for i in range(PosteriorSamples.shape[2])]
                          for j in range(N)])
        Err_histo = (abs(Histo_est-TrueParams)/(TrueParams.max(axis=0)-TrueParams.min(axis=0))).mean(axis=0)
        return {
            "Point error": Err_mean,
            "Coverage": Coverage,
            "Width": Width,
            "SignalRMSE": SignalRMSE,
            "SignalCorr": SignalCorr,
            "Point error (mode)": Err_histo
        }
    else:
        return {
            "Point error": Err_mean,
            "Coverage": Coverage,
            "Width": Width,
            "SignalRMSE": SignalRMSE,
            "SignalCorr": SignalCorr
        }

