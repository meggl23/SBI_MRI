
import h5py
import torch
from sbi.utils import BoxUniform

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from pathlib import Path

plt.style.use("seaborn-v0_8")

def make_data_folder(filename):
    """

    Ensure the data directory exists and return the filename inside it.

    """

    data_dir = Path("saved_data")

    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir / filename


def save_h5(filename, Params, S, S_raw, snr, parameter_list, compartments):
    """
    Save simulated parameters, signals, and metadata to an HDF5 file.

    If no filename is provided, a filename is generated automatically from
    the model compartment types and the current timestamp.

    Parameters
    ----------
    filename : str or None
        Output HDF5 filename. If ``None``, a filename is generated
        automatically.

    Params : array-like
        Simulated model parameters to store.

    S : array-like
        Processed or noisy simulated signals to store.

    S_raw : array-like
        Raw simulated signals to store.

    snr : float or None
        Signal-to-noise ratio associated with the simulations. If ``None``,
        the stored numerical value is -1 and an additional attribute records
        that the original value was ``None``.

    parameter_list : sequence of str
        Names of the model parameters.

    compartments : dict
        Dictionary containing the model compartment objects. Compartment class
        names are used to construct the automatic filename when ``filename``
        is ``None``.

    Returns
    -------
    None
        The function writes the data to disk and does not return a value.

    Notes
    -----
    Automatically generated filenames have the form
    ``<model_name>_<YYYYMMDD_HHMM>.h5``.
    """

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

    filename = make_data_folder(filename)
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
    """
    Load simulated parameters, signals, and metadata from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file created by ``save_h5``.

    Returns
    -------
    Params : np.ndarray
        Stored model parameters.

    S : np.ndarray
        Stored processed or noisy signals.

    S_raw : np.ndarray
        Stored raw signals.

    parameter_names : list of str
        Names of the model parameters.

    snr : float or None
        Stored signal-to-noise ratio. Returns ``None`` if the original SNR
        was stored as ``None``.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    """

    if "saved_data/" not in filename: filename= make_data_folder(filename)

    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(
            f"HDF5 file does not exist: {filename.resolve()}"
        )

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
def histogram_mode(data, bins=50):
    """
    Estimate the mode of one-dimensional data using a histogram.

    The mode is approximated as the midpoint of the histogram bin containing
    the largest number of samples.

    Parameters
    ----------
    data : array-like
        Input samples used to estimate the mode.

    bins : int or sequence, optional
        Number of histogram bins or explicit bin edges passed to
        ``numpy.histogram``. Default is 50.

    Returns
    -------
    mode : float
        Midpoint of the histogram bin with the highest sample count.

    """

    # Calculate the histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Find the bin with the maximum count (highest frequency)
    max_bin_index = np.argmax(counts)
    
    # Calculate the mode as the midpoint of the bin with the highest count
    mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    
    return mode

def Prep_data(P, S, Names, Verbose=True):
    """
    Prepare parameter and signal arrays for SBI training or inference.

    The first parameter is removed because one compartment fraction is assumed
    to be redundant and can be recovered from the remaining fractions. The
    resulting arrays are converted to PyTorch tensors and broad uniform prior
    bounds are constructed from the parameter ranges.

    Parameters
    ----------
    P : array-like
        Parameter array with shape ``(n_samples, n_parameters)``.

    S : array-like
        Signal or observation array corresponding to ``P``.

    Names : sequence of str
        Parameter names corresponding to the columns of ``P``.

    Verbose : bool, optional
        If ``True``, print which redundant parameter is being removed.
        Default is ``True``.

    Returns
    -------
    Par : torch.Tensor
        Parameter tensor after removing the first parameter column.

    Obs : torch.Tensor
        Observation tensor constructed from ``S``.

    Names_cut : sequence of str
        Parameter names after removing the first entry.

    prior_bounds : BoxUniform
        Uniform prior distribution constructed from the observed parameter
        ranges.

    """

    if Verbose: print(f'As one of the fractions is redundant we will remove it (1-sum of the rest). We are removing: {Names[0]}')

    Par = torch.tensor(P[:,1:]).float()
    Obs = torch.tensor(S).float()
    Names_cut = Names[1:]

    # Training Ranges
    low = Par.min(axis=0)[0] - 10*torch.sign(Par.min(axis=0)[0])*Par.min(axis=0)[0]
    low = np.clip(low,low,-1)
    high = Par.max(axis=0)[0] + 10*Par.max(axis=0)[0]

    prior_bounds = BoxUniform(low=low, high=high)

    return Par, Obs, Names_cut, prior_bounds

def save_posterior(output, filename=None, compression_level=4, return_dist=False):
    """
    Save posterior inference results to a compressed HDF5 file.

    Parameters
    ----------
    output : array-like
        Posterior samples or posterior summary values to store.

    filename : str or None, optional
        Output filename. If ``None``, a timestamped filename is generated
        automatically. Default is ``None``.

    compression_level : int, optional
        Gzip compression level used for the HDF5 dataset. Default is 4.

    return_dist : bool, optional
        Indicates whether ``output`` represents a full posterior distribution.
        When ``False``, ``"_mean"`` is appended to the automatically generated
        filename. Default is ``False``.

    Returns
    -------
    None
        The function writes the posterior data to disk and does not return a
        value.

    Notes
    -----
    The posterior is stored in the HDF5 dataset named ``"posterior"`` using
    gzip compression and byte shuffling.
    """

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        if( not return_dist): timestamp = timestamp + ('_mean')
        filename = f"Posterior_{timestamp}.h5"

    filename= make_data_folder(filename)

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
    Load posterior inference results from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to an HDF5 file containing a dataset named ``"posterior"``.

    Returns
    -------
    output : np.ndarray
        Stored posterior samples or posterior summary array.
    """
    if "saved_data/" not in filename: filename = make_data_folder(filename)

    filename = Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(
            f"Posterior data file does not exist: {filename.resolve()}"
        )


    with h5py.File(filename, "r") as f:
        output = f["posterior"][:]

    return output

def Add_back_fraction(Values, Names, Model):
    """
    Reconstruct a redundant compartment fraction and prepend it to the
    parameter array.

    If fraction parameters are present, the omitted fraction is calculated as
    one minus the sum of all explicitly represented fraction parameters.

    Parameters
    ----------
    Values : array-like
        Parameter values. The final axis must correspond to the entries in
        ``Names``.

    Names : sequence of str
        Names of the parameters represented in ``Values``. Fraction parameters
        are identified by names containing ``"_f"``.

    Model : object
        Model containing the complete ``parameter_list``.

    Returns
    -------
    Values_full : np.ndarray
        Parameter array with the reconstructed fraction prepended along the
        final axis.

    parameter_list : sequence of str
        Complete parameter list obtained from ``Model.parameter_list``.

    Raises
    ------
    ValueError
        If no fraction parameter is present in ``Names``.

    Notes
    -----
    The reconstructed fraction is calculated as

    ``1 - sum(explicit fraction parameters)``.
    """

    if(np.any(['_f' in N for N in Names])):
        f0 = 1 - np.sum(
            Values[..., ['_f' in N for N in Names]],
            axis=-1
        )

        return np.concatenate([f0[..., None], Values],axis=-1), Model.parameter_list
    else:
        raise ValueError(f"This model only had 1 compartment - no point adding back the fraction!")

def Plot_parameter_map(Values,Model,parameter,slices=[],cmap='hot',data_range=[],save=False,Filename=''):
    """
    Plot two-dimensional or three-dimensional maps of a model parameter.

    For two-dimensional parameter maps, a single image is displayed. For
    three-dimensional volumes, orthogonal slices through the volume are shown.

    Parameters
    ----------
    Values : np.ndarray
        Array containing spatially resolved model parameter values.

    Model : object
        Model defining the available parameters and providing
        ``_get_parameter_index``.

    parameter : str
        Parameter to visualize. It must normally be present in
        ``Model.parameter_list``. Derived parameters ``"MD"`` and ``"FA"`` are
        also accepted.

    slices : sequence of int, optional
        Slice indices used for three-dimensional data. If empty, the central
        slice along each dimension is selected automatically.

    cmap : str, optional
        Matplotlib colormap used for the image. Default is ``"hot"``.

    data_range : sequence of float, optional
        Two-element sequence specifying the minimum and maximum displayed
        values. If empty, the range is determined from the data.

    save : bool, optional
        If ``True``, save the resulting figure as a PDF. Default is ``False``.

    Filename : str, optional
        Filename argument associated with figure saving. Default is an empty
        string.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created Matplotlib figure.

    ax : matplotlib.axes.Axes or np.ndarray
        Axes containing the displayed parameter map or maps.

    Raises
    ------
    ValueError
        If ``parameter`` is not available in the model and is not ``"MD"`` or
        ``"FA"``.
    """

    Val = parameter_selector(Values,Model,parameter)

    if(len(data_range) == 0): data_range = [np.nanmin(Val),np.nanmax(Val)]
    
    if(Val.ndim == 2):
        fig, ax = plt.subplots()
        plt.imshow(Val,cmap=cmap,vmin = data_range[0],vmax = data_range[1])
        ax.axis('off')

    elif(Val.ndim == 3):
        if(len(slices) == 0):
            slices = [r//2 for r in Val.shape]
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(Val[slices[0]],cmap=cmap,vmin = data_range[0],vmax = data_range[1])
        ax[0].axis('off')
        ax[1].imshow(Val[:,slices[1]],cmap=cmap,vmin = data_range[0],vmax = data_range[1])
        ax[1].axis('off')
        ax[2].imshow(Val[...,slices[2]],cmap=cmap,vmin = data_range[0],vmax = data_range[1])
        ax[2].axis('off')
    fig.suptitle(parameter)
    if(save):
        save_figure(fig,parameter)
    return fig,ax

def Plot_parameter_maps(Values,Model,parameters,slices=[],cmap='hot',data_range=[],save=False,Filename=''):
    """
    Plot multiple model parameter maps sequentially.

    Each parameter is passed to ``Plot_parameter_map`` and its resulting figure
    is displayed.

    Parameters
    ----------
    Values : np.ndarray
        Array containing spatially resolved model parameter values.

    Model : object
        Model defining the available parameters.

    parameters : sequence of str
        Names of the parameters to visualize.

    slices : sequence of int, optional
        Slice indices used for three-dimensional parameter volumes. If empty,
        central slices are selected automatically.

    cmap : str, optional
        Matplotlib colormap used for all parameter maps. Default is ``"hot"``.

    data_range : sequence of float, optional
        Two-element display range applied to all parameter maps. If empty,
        each parameter determines its range from its own data.

    save : bool, optional
        If ``True``, save each generated figure. Default is ``False``.

    Filename : str, optional
        Filename argument passed to ``Plot_parameter_map``.

    Returns
    -------
    None
        The function displays each generated figure and does not explicitly
        return a value.
    """
    for p in parameters:
        fig,ax = Plot_parameter_map(Values,Model, p,slices = slices,cmap = cmap, data_range = data_range,save=save,Filename=Filename)
        plt.show()

def parameter_selector(Values, Model, parameter):
    """
    Extract a selected model parameter from a parameter array.

    Parameters
    ----------
    Values : np.ndarray
        Array containing model parameter values. The final axis is assumed to
        correspond to model parameters.

    Model : object
        Model providing ``parameter_list`` and ``_get_parameter_index``.

    parameter : str
        Parameter to select. Parameters in ``Model.parameter_list`` are
        extracted from the final axis. Derived parameters ``"MD"`` and
        ``"FA"`` are also accepted.

    Returns
    -------
    Val : np.ndarray
        Selected parameter values. For ``"MD"`` or ``"FA"``, a copy of the
        input array is returned for subsequent derived-parameter processing.

    Raises
    ------
    ValueError
        If the requested parameter is not present in
        ``Model.parameter_list`` and is not ``"MD"`` or ``"FA"``.
    """
    if(parameter not in Model.parameter_list): 
        if parameter == "MD" or parameter == "FA":
            Val = np.copy(Values)
        else:
            raise ValueError("Invalid parameter choice (check model.parmater_list or use MD or FA)")
    else:       
        Val = Values[..., Model._get_parameter_index(parameter)]

    return Val
def Widget_parameter_map(Values,Model,parameter,view='ax',cmap='hot',data_range=[]):
    """
    Display an interactive slice viewer for a three-dimensional parameter map.

    An integer slider is used to move through slices along the selected
    anatomical viewing direction. The displayed image and colorbar are updated
    whenever the slider position changes.

    Parameters
    ----------
    Values : np.ndarray
        Array containing spatially resolved model parameter values.

    Model : object
        Model defining the available parameters.

    parameter : str
        Parameter to display.

    view : {'sag', 'cor', 'ax'}, optional
        Viewing direction used for slicing the volume:

        - ``"sag"`` selects slices along axis 0.
        - ``"cor"`` selects slices along axis 1.
        - ``"ax"`` selects slices along axis 2.

        Default is ``"ax"``.

    cmap : str, optional
        Matplotlib colormap used for visualization. Default is ``"hot"``.

    data_range : sequence of float, optional
        Two-element sequence defining the displayed minimum and maximum
        values. If empty, the limits are determined from the parameter data.

    Returns
    -------
    None
        The function displays an interactive Jupyter widget and does not
        explicitly return a value.

    Raises
    ------
    ValueError
        If ``view`` is not one of ``"sag"``, ``"cor"``, or ``"ax"``.

    Notes
    -----
    This function is intended for notebook environments supporting
    ``ipywidgets`` and IPython ``display``.
    """
    # Extract the 3D parameter volume

    Val = parameter_selector(Values,Model,parameter)

    if(len(data_range) == 0): data_range = [np.nanmin(Val),np.nanmax(Val)]

    # Decide which axis the slider should iterate over
    if view == 'sag':
        axis = 0
    elif view == 'cor':
        axis = 1
    elif view == 'ax':
        axis = 2
    else:
        raise ValueError("Chose invalid viewing option - please choose: sag, cor or ax")

    maxview = Val.shape[axis]

    # Helper to extract the appropriate 2D slice
    def get_slice(i):
        if view == 'sag':
            return Val[i, :, :]
        elif view == 'cor':
            return Val[:, i, :]
        elif view == 'ax':
            return Val[:, :, i]

    slider = widgets.IntSlider(
        value=maxview // 2,
        min=0,
        max=maxview - 1,
        step=1,
        description='Slice:',
        continuous_update=True
    )

    output = widgets.Output()

    def update(change=None):
        i = slider.value
        data = get_slice(i)

        with output:
            output.clear_output(wait=True)

            fig, ax = plt.subplots(figsize=(6, 6))

            if len(data_range) == 2:
                im = ax.imshow(
                    data,
                    cmap=cmap,
                    vmin=data_range[0],
                    vmax=data_range[1]
                )
            else:
                im = ax.imshow(
                    data,
                    cmap=cmap
                )

            ax.set_title(f'{parameter} | {view} | slice {i}')
            ax.axis('off')

            fig.colorbar(im, ax=ax)
            plt.show()

    slider.observe(update, names='value')

    update()
    display(widgets.VBox([slider, output]))


def save_figure(fig, parameter, filename=None):
    """
    Save a Matplotlib figure as a transparent PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.

    parameter : str
        Parameter name used when automatically generating the filename.

    filename : str or None, optional
        Output filename without the ``.pdf`` extension. If ``None``, a
        timestamped filename is generated automatically.

    Returns
    -------
    None
        The function writes the figure to disk and does not return a value.

    Notes
    -----
    Automatically generated filenames have the form
    ``<parameter>_<YYYYMMDD_HHMM>.pdf``. The figure is saved with a transparent
    background and a tight bounding box.
    """

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{parameter}_{timestamp}"

    filename= make_data_folder(filename)
    fig.savefig(f"{filename}.pdf",format='pdf',transparent=True,bbox_inches='tight')

    print(f"Figure saved to '{filename}.pdf'")




