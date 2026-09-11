from abc import ABC, abstractmethod
from joblib import Parallel, delayed

from scipy.special import j0, jv
from scipy.special import spherical_jn
from scipy.optimize import bisect
from scipy.stats import poisson
import numpy as np

import Helpers

import tqdm

#----------------Generic Compartment---------------------
class Compartment(ABC):
    """
    Abstract base class for diffusion-model compartments.

    A compartment defines how model parameters are sampled, how corresponding
    signals are simulated, and which parameters belong to the compartment.
    Concrete compartment classes must implement ``simulation``, ``sample``,
    and ``build_parameter_list``.

    Parameters
    ----------
    gtab : object
        Gradient table or acquisition description used by the compartment
        during signal simulation.
    """
    def __init__(self,gtab):
        """
        Initialize the compartment.

        Parameters
        ----------
        gtab : object
            Gradient table or acquisition description used during simulation.
        """
        self.gtab = gtab
    
    @abstractmethod
    def simulation(self,params):
        """
        Simulate signals from compartment parameters.

        Parameters
        ----------
        params : array-like
            Parameter values used to generate the simulated signals.

        Returns
        -------
        signals : array-like
            Simulated signals corresponding to ``params``.
        """
        pass

    @abstractmethod
    def sample(self,N=1,rng = None):
        """
        Sample parameter values from the compartment parameter distribution.

        Parameters
        ----------
        N : int, optional
            Number of parameter samples to generate. Default is 1.

        rng : numpy.random.Generator or None, optional
            Random number generator used for sampling. If ``None``, the
            implementation may create its own generator. Default is ``None``.

        Returns
        -------
        params : array-like
            Sampled compartment parameters.
        """
        pass
        
    def sample_and_simulate(self, N=1, rng = None):
        """
        Sample compartment parameters and simulate their corresponding signals.

        Parameters are first generated using ``sample`` and are then passed to
        ``simulation`` to obtain the associated signals.

        Parameters
        ----------
        N : int, optional
            Number of parameter sets and signals to generate. Default is 1.

        rng : numpy.random.Generator or None, optional
            Random number generator used for parameter sampling. If ``None``,
            a new NumPy random generator is created. Default is ``None``.

        Returns
        -------
        params : array-like
            Sampled compartment parameters.

        signals : array-like
            Simulated signals corresponding to the sampled parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        params = self.sample(N,rng = rng)
        signals = self.simulation(params)
        return params, signals
        
    @abstractmethod
    def build_parameter_list(self):
        """
        Construct the list of parameters used by the compartment.

        Returns
        -------
        parameter_list : sequence of str
            Names of the parameters required by the compartment.
        """
        pass

    @staticmethod
    def _check_range(name, min_val, max_val):
        """
        Validate that a parameter range is ordered correctly.

        Parameters
        ----------
        name : str
            Name of the parameter being checked.

        min_val : float
            Minimum allowed value.

        max_val : float
            Maximum allowed value.

        Returns
        -------
        None
            The function only validates the supplied range.

        Raises
        ------
        ValueError
            If ``min_val`` is greater than ``max_val``.
        """
        if min_val > max_val:
            raise ValueError(
                f"{name}: minimum ({min_val}) must not exceed maximum ({max_val})."
            )    
        
    def sample_and_simulate_parallel(self, N=1, n_jobs=-1, chunk_size=1000,rng = None):
        """
        Sample compartment parameters and simulate signals in parallel.

        Parameters are sampled once and divided into chunks. Each chunk is
        simulated independently using parallel workers, after which the
        resulting signal arrays are concatenated.

        Parameters
        ----------
        N : int, optional
            Number of parameter sets and signals to generate. Default is 1.

        n_jobs : int, optional
            Number of parallel workers passed to ``joblib.Parallel``.
            A value of -1 uses all available workers. Default is -1.

        chunk_size : int, optional
            Maximum number of parameter sets simulated by each parallel task.
            Default is 1000.

        rng : numpy.random.Generator or None, optional
            Random number generator used for parameter sampling. If ``None``,
            a new NumPy random generator is created. Default is ``None``.

        Returns
        -------
        params : array-like
            Sampled compartment parameters.

        signals : np.ndarray
            Simulated signals for all sampled parameter sets. Signal chunks
            produced by the parallel workers are concatenated along the first
            axis.
        """

        if rng is None:
            rng = np.random.default_rng()

        params = self.sample(N,rng = rng)
    
        chunks = [
            params[i:i + chunk_size]
            for i in range(0, N, chunk_size)
        ]
    
        signals = Parallel(n_jobs=n_jobs)(
            delayed(self.simulation)(chunk)
            for chunk in chunks
        )
    
        signals = np.vstack(signals)
    
        return params, signals

#----------------DTI Compartment---------------------
class DTI(Compartment):
    """
    Diffusion Tensor Imaging (DTI) compartment.

    This compartment represents diffusion using a symmetric 3 × 3 diffusion
    tensor. Random tensors are generated from sampled mean diffusivity (MD) and
    fractional anisotropy (FA) values, combined with a random three-dimensional
    orientation.

    Parameters
    ----------
    gtab : object
        Gradient table containing acquisition gradient directions and
        b-values.

    MD_min : float, optional
        Minimum mean diffusivity used when sampling random tensors.
        Default is 0.

    MD_max : float, optional
        Maximum mean diffusivity used when sampling random tensors.
        Default is 0.005.

    FA_min : float, optional
        Minimum fractional anisotropy used when sampling random tensors.
        Default is 0.

    FA_max : float, optional
        Maximum fractional anisotropy used when sampling random tensors.
        Default is 0.999.

    Attributes
    ----------
    MD_range : list of float
        Minimum and maximum MD values used for sampling.

    FA_range : list of float
        Minimum and maximum FA values used for sampling.

    parameter_names : list of str
        Names of the six independent diffusion-tensor components.

    DTI_gen_max_tries : int
        Maximum number of attempts made when generating a positive-definite
        diffusion tensor.

    clip : bool
        If ``True``, negative or very small tensor eigenvalues are clipped
        before tensors are used.
    """

    def __init__(self,gtab,MD_min = 0, MD_max = 0.005, FA_min = 0, FA_max = 0.999):
        super().__init__(gtab)
        self.MD_range = [MD_min,MD_max]
        self.FA_range = [FA_min,FA_max]

        self._check_range("MD", MD_min, MD_max)
        self._check_range("FA", FA_min, FA_max)

        self.DTI_gen_max_tries = 10000

        self.build_parameter_list()

        self.clip = True
    
    def simulation(self, tensors, clip = True):
        """
        Simulate diffusion-weighted signals from diffusion tensors.

        Each tensor is converted to a 3 × 3 symmetric matrix and used to
        calculate the apparent diffusion coefficient along every gradient
        direction. The signal is then computed using the mono-exponential
        diffusion model.

        Parameters
        ----------
        tensors : array-like
            Diffusion tensors in flattened six-component representation with
            shape ``(N, 6)``.

        clip : bool, optional
            Intended control for eigenvalue clipping. The current
            implementation uses the instance attribute ``self.clip`` to decide
            whether clipping is performed. Default is ``True``.

        Returns
        -------
        signals : np.ndarray
            Simulated diffusion-weighted signals with shape
            ``(N, n_gradients)``.

        Raises
        ------
        ValueError
            If the converted tensors do not have shape ``(N, 3, 3)``.

        Notes
        -----
        The signal is calculated as

        ``exp(-b * g.T @ D @ g)``,

        where ``D`` is the diffusion tensor, ``g`` is the gradient direction,
        and ``b`` is the corresponding b-value.
        """

        if(self.clip):
            tensors = np.array([DTI.clip_negative_eigenvalues(t) for t in tensors])
        tensors = np.asarray([DTI.flat_to_tens(t) for t in tensors])
    
        if tensors.shape[1:] != (3, 3):
            raise ValueError("tensors must have shape (N, 3, 3)")
    
        gradients = self.gtab.bvecs.reshape(-1, 3)
        bvals = self.gtab.bvals.reshape(-1)
    
        adc = np.einsum("ij,njk,ik->ni", gradients, tensors, gradients)
    
        return np.exp(-bvals[None, :] * adc)
        
    def sample(self, N=1, rng=None):
        """
        Sample random diffusion tensors from the configured MD and FA ranges.

        Mean diffusivity and fractional anisotropy are sampled independently
        from uniform distributions. A positive-definite tensor with the
        requested MD and FA is then generated with a random orientation.

        Parameters
        ----------
        N : int, optional
            Number of diffusion tensors to generate. Default is 1.

        rng : numpy.random.Generator or None, optional
            Random number generator used for sampling. If ``None``, a new
            NumPy random generator is created. Default is ``None``.

        Returns
        -------
        tensors : np.ndarray
            Sampled diffusion tensors in flattened six-component
            representation with shape ``(N, 6)``.
        """

        if rng is None:
            rng = np.random.default_rng()

        MD_samples = rng.uniform(*self.MD_range, size=N)

        FA_samples = rng.uniform(*self.FA_range, size=N)

        tensors = np.array([
            DTI.tens_to_flat(self.random_diffusion_tensor(md, fa, rng))
            for md, fa in zip(MD_samples, FA_samples)
        ])

        if(self.clip):
            tensors = np.array([DTI.clip_negative_eigenvalues(t) for t in tensors])

        return tensors

    def set_MD_FA_priors(self):
        """
        Placeholder for configuring custom MD and FA prior distributions.

        Returns
        -------
        NotImplemented
            Indicates that custom MD and FA priors are not currently
            implemented.
        """
        return NotImplemented

    def build_parameter_list(self):
        """
        Define the parameter names for the flattened diffusion tensor.

        The six independent components of the symmetric diffusion tensor are
        stored in the order ``D_xx``, ``D_xy``, ``D_yy``, ``D_xz``, ``D_yz``,
        and ``D_zz``.

        Returns
        -------
        None
            The parameter names are stored in ``self.parameter_names``.
        """
        self.parameter_names = ['D_xx','D_xy','D_yy','D_xz','D_yz','D_zz']

    @staticmethod
    def flat_to_tens(flat_tensor):
        """
        Convert a flattened diffusion tensor into a symmetric 3 × 3 matrix.

        Parameters
        ----------
        flat_tensor : array-like
            Six tensor components ordered as
            ``[D_xx, D_xy, D_yy, D_xz, D_yz, D_zz]``.

        Returns
        -------
        tensor : np.ndarray
            Symmetric diffusion tensor with shape ``(3, 3)``.
        """
        xx, xy, yy, xz, yz, zz = flat_tensor
    
        return np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ])
    
    @staticmethod
    def tens_to_flat(tensor):
        """
        Convert a symmetric 3 × 3 diffusion tensor to six-component form.

        Parameters
        ----------
        tensor : array-like
            Diffusion tensor with shape ``(3, 3)``.

        Returns
        -------
        flat_tensor : np.ndarray
            Flattened tensor ordered as
            ``[D_xx, D_xy, D_yy, D_xz, D_yz, D_zz]``.
        """
        return np.array([
            tensor[0, 0],
            tensor[0, 1],
            tensor[1, 1],
            tensor[0, 2],
            tensor[1, 2],
            tensor[2, 2],
        ])

    @staticmethod
    def clip_negative_eigenvalues(tensor):
        """
        Enforce positive diffusion-tensor eigenvalues.

        The tensor is eigendecomposed and every eigenvalue smaller than
        ``1e-5`` is replaced by ``1e-5``. The tensor is then reconstructed
        using the original eigenvectors.

        Parameters
        ----------
        tensor : array-like
            Diffusion tensor represented either as six flattened components
            with shape ``(6,)`` or as a matrix with shape ``(3, 3)``.

        Returns
        -------
        clipped_tensor : np.ndarray
            Tensor with clipped eigenvalues. The output representation matches
            the input representation.
        """
        tensor = np.asarray(tensor)

        is_flat = tensor.shape == (6,)

        if is_flat:
            matrix = DTI.flat_to_tens(tensor)
        else:
            matrix = tensor

        # Symmetric eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Clip eigenvalues
        clipped_eigenvalues = np.maximum(eigenvalues, 1e-5)

        # Reconstruct matrix
        clipped_matrix = (
            eigenvectors
            @ np.diag(clipped_eigenvalues)
            @ eigenvectors.T
        )

        if is_flat:
            return DTI.tens_to_flat(clipped_matrix)

        return clipped_matrix

    def random_diffusion_tensor(self,MD, FA,rng):
        """
        Generate a random positive-definite diffusion tensor with specified
        mean diffusivity and fractional anisotropy.

        Eigenvalues satisfying the requested MD and FA are generated using a
        random angular parameter. A uniformly distributed random rotation is
        then applied to produce an arbitrary three-dimensional orientation.

        Parameters
        ----------
        MD : float
            Desired mean diffusivity. Must be strictly positive.

        FA : float
            Desired fractional anisotropy. Must satisfy ``0 <= FA < 1``.

        rng : numpy.random.Generator
            Random number generator used to generate eigenvalues and
            orientation.

        Returns
        -------
        tensor : np.ndarray
            Positive-definite diffusion tensor with shape ``(3, 3)``.

        Raises
        ------
        ValueError
            If ``MD <= 0`` or if ``FA`` is outside the interval
            ``[0, 1)``.

        RuntimeError
            If a set of positive eigenvalues cannot be generated within
            ``DTI_gen_max_tries`` attempts.
        """

        if MD <= 0:
            raise ValueError("MD must be positive")
    
        if not (0 <= FA < 1):
            raise ValueError("FA must satisfy 0 <= FA < 1")
    
        X = (3 * FA**2 * MD**2) / (1.5 - FA**2)
    
        for _ in range(self.DTI_gen_max_tries):
            theta = rng.uniform(0, 2 * np.pi)
    
            a = np.sqrt(X / 6) * np.cos(theta)
            b = np.sqrt(X / 2) * np.sin(theta)
    
            l1 = MD + a + b
            l2 = MD + a - b
            l3 = MD - 2 * a
    
            lambdas = np.array([l1, l2, l3])
    
            if np.all(lambdas > 0):
                Q = self.random_SO3(rng)
                return Q @ np.diag(lambdas) @ Q.T
    
        raise RuntimeError("Could not generate positive eigenvalues")
    
    @staticmethod
    def random_SO3(rng):
        """
        Generate a random three-dimensional rotation matrix.

        A random Gaussian matrix is orthogonalized using QR decomposition and
        corrected so that the resulting matrix belongs to the special
        orthogonal group SO(3).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to generate the initial matrix.

        Returns
        -------
        Q : np.ndarray
            Random rotation matrix with shape ``(3, 3)`` and determinant +1.
        """

        M = rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
    
        # make Q uniformly distributed over SO(3)
        Q = Q @ np.diag(np.sign(np.diag(R)))
    
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
    
        return Q

    @staticmethod
    def EvaluateTensor(tensors, clip_negative=False, eps=1e-5):
        """
        Compute mean diffusivity, fractional anisotropy, eigenvalues, and
        eigenvectors from diffusion tensors.

        Parameters
        ----------
        tensors : array-like
            Diffusion tensors with shape ``(..., 6)`` or ``(..., 3, 3)``.

        clip_negative : bool, optional
            If ``True``, eigenvalues smaller than ``eps`` are replaced by
            ``eps`` before calculating MD and FA. Default is ``False``.

        eps : float, optional
            Minimum eigenvalue used when ``clip_negative=True``.
            Default is ``1e-5``.

        Returns
        -------
        MD_FA_values : np.ndarray
            Array with shape ``original_shape + (2,)``.
            ``[..., 0]`` contains MD and ``[..., 1]`` contains FA.

        eigenvalues : np.ndarray
            Eigenvalues with shape ``original_shape + (3,)``.

        eigenvectors : np.ndarray
            Eigenvectors with shape ``original_shape + (3, 3)``.
            Eigenvectors are stored column-wise, following
            ``np.linalg.eigh`` conventions.

        Raises
        ------
        ValueError
            If the input does not have shape ``(..., 6)`` or
            ``(..., 3, 3)``.
        """

        tensors = np.asarray(tensors)

        # Case 1: flattened symmetric tensor representation (..., 6)
        if tensors.shape[-1:] == (6,):
            original_shape = tensors.shape[:-1]

            flat = tensors.reshape(-1, 6)

            mats = np.array([
                DTI.flat_to_tens(x)
                for x in flat
            ])

            tensors = mats.reshape(original_shape + (3, 3))

        # Case 2: full tensor representation (..., 3, 3)
        elif tensors.shape[-2:] == (3, 3):
            original_shape = tensors.shape[:-2]

        else:
            raise ValueError(
                "Expected shape (..., 6) or (..., 3, 3)."
            )

        # Flatten spatial dimensions
        flat_tensors = tensors.reshape(-1, 3, 3)
        n_tensors = len(flat_tensors)

        # Default everything to zero
        MD = np.zeros(n_tensors, dtype=float)
        FA = np.zeros(n_tensors, dtype=float)

        eigenvalues = np.zeros((n_tensors, 3), dtype=float)
        eigenvectors = np.zeros((n_tensors, 3, 3), dtype=float)

        for i, tensor in enumerate(flat_tensors):

            # Invalid tensor -> everything remains zero
            if not np.all(np.isfinite(tensor)):
                continue

            try:
                evals, evecs = np.linalg.eigh(tensor)
            except np.linalg.LinAlgError:
                continue

            if (
                not np.all(np.isfinite(evals))
                or not np.all(np.isfinite(evecs))
            ):
                continue

            if clip_negative:
                evals = np.maximum(evals, eps)

            md = np.mean(evals)

            denom = np.sum(evals**2)
            numer = 3 * np.sum((evals - md)**2)

            MD[i] = md
            eigenvalues[i] = evals
            eigenvectors[i] = evecs

            if denom > 0:
                FA[i] = np.sqrt(numer / (2 * denom))

        # Restore original dimensions
        MD = MD.reshape(original_shape)
        FA = FA.reshape(original_shape)

        eigenvalues = eigenvalues.reshape(original_shape + (3,))
        eigenvectors = eigenvectors.reshape(original_shape + (3, 3))

        MD_FA_values = np.stack((MD, FA), axis=-1)

        return MD_FA_values, eigenvalues, eigenvectors

#----------------Stick Compartment---------------------
class Sticks(Compartment):
    """
    Restricted anisotropic diffusion compartment with optional orientation
    dispersion and axon-size modelling.

    The compartment models diffusion using an oriented cylindrical geometry
    with separate parallel and perpendicular diffusivities. Fibre orientation
    is parameterized by spherical angles. Optional Watson orientation
    dispersion and several radius models are supported.

    Parameters
    ----------
    gtab : object
        Gradient table containing gradient directions, b-values, and diffusion
        timing information.

    D_par_min : float, optional
        Minimum parallel diffusivity used for sampling. Default is 0.

    D_par_max : float, optional
        Maximum parallel diffusivity used for sampling. Default is 5e-3.

    D_perp_min : float, optional
        Minimum perpendicular diffusivity used for sampling. Default is 0.

    D_perp_max : float, optional
        Maximum perpendicular diffusivity used for sampling. Default is 5e-3.

    Dispersion : bool, optional
        If ``True``, include orientation dispersion through a Watson
        concentration parameter ``kappa``. Default is ``False``.

    kappa_min : float, optional
        Minimum Watson concentration parameter used for sampling.
        Default is 0.

    kappa_max : float, optional
        Maximum Watson concentration parameter used for sampling.
        Default is 10.

    Size : {'Infer', 'Distribution', 'Single'}, optional
        Strategy used for compartment radius modelling.

        ``'Infer'``
            Treat radius as an inferred model parameter.

        ``'Distribution'``
            Use a predefined discrete radius distribution.

        ``'Single'``
            Use a single fixed radius.

        Default is ``'Infer'``.

    size_min : float, optional
        Minimum radius used when ``Size='Infer'``. Default is 1e-4.

    size_max : float, optional
        Maximum radius used when ``Size='Infer'``. Default is 0.005.
    """
    def __init__(self,gtab, D_par_min = 0, D_par_max = 5e-3, D_perp_min = 0,D_perp_max = 5e-3,
                Dispersion=False,kappa_min = 0, kappa_max = 10, Size = 'Infer',size_min=1e-4,size_max=0.005):
        super().__init__(gtab)

        self.D_par_range  = [D_par_min,D_par_max]
        self._check_range("D_par", D_par_min, D_par_max)
        self.D_perp_range = [D_perp_min,D_perp_max]
        self._check_range("D_perp", D_perp_min, D_perp_max)
        self.dispersion = False
        self.size  = Size

        allowed = {"Infer", "Distribution", "Single"}

        if Size not in allowed:
            raise ValueError(
                f"comp_type must be one of {allowed}, got '{Size}'"
            )

        if Dispersion:
            self.kappa_range = [kappa_min,kappa_max]
            self._check_range("kappa", kappa_min, kappa_max)
            self.dispersion  = True
        if Size == 'Infer':
            if len(np.unique(self.gtab.big_delta))  < 2:
                print('*****WARNING: You are trying to fit size with only 1 diffusion time - proceed with caution!*****')
            self.size_range = [size_min,size_max]
            self._check_range("size", size_min, size_max)
            
        n_roots = 15
        self.Bessel_roots = np.array(self.j1prime_zeros(n_roots, x_max=10e6, step=0.01))

        self.bins_theta=10
        self.bins_phi=20

        self.build_parameter_list()
        
    def build_parameter_list(self):
        """
        Construct the list of parameters used by the compartment.

        The base parameterization contains fibre orientation, parallel
        diffusivity, and perpendicular diffusivity. Additional parameters are
        included when orientation dispersion or inferred radius are enabled.

        Returns
        -------
        None
            Parameter names are stored in ``self.parameter_names``.
        """
        self.parameter_names = ['theta','phi','D_par','D_perp']
        if(self.dispersion): self.parameter_names.append('kappa')
        if(self.size == 'Infer'): self.parameter_names.append('size')
        
    def j1_derivative(self,x):
        """Derivative of J1(x) using the identity: J1'(x) = 0.5 * (J0(x) - J2(x))."""
        return 0.5 * (j0(x) - jv(2, x))

    
    def j1prime_zeros(self,n, x_max=100, step=0.1):
        """
        Find the first n positive roots of J1'(x) by scanning from x=0 to x_max.
        
        Parameters
        ----------
        n     : int
            Number of roots to find
        x_max : float
            Maximum x to search
        step  : float
            Step size for scanning sign changes
        
        Returns
        -------
        zeros : list of float
            List of the first n roots (x > 0) of J1'(x).
        """
        zeros = []
        x_vals = np.arange(0.0, x_max, step)
        
        f_prev = self.j1_derivative(x_vals[0])
        for i in range(1, len(x_vals)):
            f_curr = self.j1_derivative(x_vals[i])
            # Check for a sign change in [x_vals[i-1], x_vals[i]]
            if f_prev * f_curr < 0:
                root = bisect(self.j1_derivative, x_vals[i-1], x_vals[i])
                zeros.append(root)
                if len(zeros) == n:
                    break
            f_prev = f_curr
        
        return zeros

    def sample(self,N,rng = None):
        """
        Sample random parameter sets for the Sticks compartment.

        Parallel and perpendicular diffusivities are sampled uniformly from
        their configured ranges. Fibre orientations are sampled uniformly on
        the sphere and converted to spherical coordinates.

        Optional dispersion and size parameters are appended when enabled.

        Parameters
        ----------
        N : int
            Number of parameter sets to generate.

        rng : numpy.random.Generator or None, optional
            Random number generator used for sampling. If ``None``, a new
            NumPy generator is created. Default is ``None``.

        Returns
        -------
        params : np.ndarray
            Sampled parameter array with shape
            ``(N, n_parameters)``.
        """

        if rng is None:
            rng = np.random.default_rng()

        D_par_samples = rng.uniform(*self.D_par_range, size=N)
        D_perp_samples = rng.uniform(*self.D_perp_range, size=N)


        V = rng.standard_normal((N, 3))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        Angs = np.array([self.SpherAng(v) for v in V])

        params = np.column_stack([Angs,D_par_samples,D_perp_samples])
        
        if(self.dispersion):
            kappa_samples = rng.uniform(*self.kappa_range, size=N)
            params = np.column_stack([params, kappa_samples])

        if(self.size == 'Infer'):
            size_samples = rng.uniform(*self.size_range, size=N)
            params = np.column_stack([params, size_samples])
        return params

    def SpherAng(self,vector):

        """
        Convert a three-dimensional direction vector to spherical angles.

        Vectors pointing into the lower hemisphere are flipped so that the
        resulting orientation lies in the upper hemisphere.

        Parameters
        ----------
        vector : array-like
            Three-dimensional direction vector.

        Returns
        -------
        theta : float
            Polar angle in radians.

        phi : float
            Azimuthal angle in radians.

        Notes
        -----
        A zero-length vector is assigned ``theta = 0`` and ``phi = 0``.
        """

        if vector[2] < 0:
            vector = -vector  # Flip the vector to the top hemisphere
        x, y, z = vector
        r = np.linalg.norm(vector)
        if r == 0:
            # Degenerate vector, define angles however you like:
            return 0.0, 0.0
        
        # Polar angle in [0, pi]
        theta = np.arccos(z / r)
        # Azimuthal angle in (-pi, pi]
        phi = np.arctan2(y, x)
            
        return theta,phi

    def simulation(self, params):
        """
        Simulate diffusion-weighted signals for one or more parameter sets.

        Parameters
        ----------
        params : array-like
            Model parameters. A one-dimensional array represents one parameter
            set, while a two-dimensional array represents multiple parameter
            sets.

        Returns
        -------
        signals : np.ndarray
            Simulated diffusion-weighted signals. The first dimension
            corresponds to parameter sets.
        """

        params = np.asarray(params)
    
        if params.ndim == 1:
            return self.simulate_one(params)[None, :]
    
        return np.vstack([self.simulate_one(p) for p in params])
        
    def simulate_one(self, params):
        """
        Simulate the signal for a single Sticks parameter set.

        The model combines parallel Gaussian diffusion with restricted
        perpendicular diffusion. Optional orientation dispersion and radius
        distributions are incorporated by weighted numerical averaging.

        Parameters
        ----------
        params : array-like
            Parameters for a single compartment realization. The exact order
            depends on whether dispersion and radius inference are enabled.

        Returns
        -------
        signal : np.ndarray
            Simulated signal for each acquisition in the gradient table.

        Notes
        -----
        Restricted perpendicular diffusion is evaluated using a truncated
        Bessel-root expansion. Orientation dispersion is incorporated through
        weighted orientation bins and radius distributions are integrated
        numerically over ``SizeGen``.
        """

        if self.dispersion and self.size == 'Infer':
            theta_f, phi_f, Dpar, Dperp, kappa, size = params
        elif self.dispersion:
            theta_f, phi_f, Dpar, Dperp, kappa = params
            size = self.size
        elif self.size == 'Infer':
            theta_f, phi_f, Dpar, Dperp, size = params
            kappa = None
        else:
            theta_f, phi_f, Dpar, Dperp = params
            kappa = None
            size = self.size
    
        bvecs = np.asarray(self.gtab.bvecs, dtype=np.float64)
        bvals = np.asarray(self.gtab.bvals, dtype=np.float64)
        delta = np.asarray(self.gtab.small_delta, dtype=np.float64)
        Delta = np.asarray(self.gtab.big_delta, dtype=np.float64)

        
        V_bins, weights_bins = self.Orientation(theta_f, phi_f, kappa)
        R,wR = self.SizeGen(size)
        R2 = R**2
    
        # bin-measurement angles
        bnorm = np.linalg.norm(bvecs, axis=1)
        bnorm_safe = np.where(bnorm == 0.0, 1.0, bnorm)
    
        cosang = (V_bins @ bvecs.T) / bnorm_safe[None, :]
        cosang = np.clip(cosang, -1.0, 1.0)
    
        cos2 = cosang**2        # (Nb, M)
        sin2 = 1.0 - cos2       # (Nb, M)
    
        # Bessel roots
        br = self.Bessel_roots[:10]
        br2 = br**2
        br6 = br**6
    
        Dperp_ = Dperp
        R2_ = R2[:, None, None]       # (Nr, 1, 1)
        br2_ = br2[None, :, None]     # (1, m, 1)
        br6_ = br6[None, :, None]     # (1, m, 1)
        delta_ = delta[None, None, :] # (1, 1, M)
        Delta_ = Delta[None, None, :] # (1, 1, M)
    
        num = (
            2 * Dperp_ * br2_ * delta_ / R2_ - 2
            + 2 * np.exp(-Dperp_ * br2_ * delta_ / R2_)
            + 2 * np.exp(-Dperp_ * br2_ * Delta_ / R2_)
            - np.exp(-Dperp_ * br2_ * (Delta_ - delta_) / R2_)
            - np.exp(-Dperp_ * br2_ * (Delta_ + delta_) / R2_)
        )
    
        den = (
            Dperp_**2
            * br6_
            * (br2_ - 1.0)
            / R2_**3
        )
    
        sumR = np.sum(num / den, axis=1)  # (Nr, M)
    
        x = -2.0 * bvals[None, :] * sin2 / (
            (Delta[None, :] - delta[None, :] / 3.0) * delta[None, :]**2
        )  # (Nb, M)
    
        Y = np.exp(x[:, None, :] * sumR[None, :, :])  # (Nb, Nr, M)
    
        weighted = np.sum(Y * wR[None, :, None], axis=1)  # (Nb, M)
    
        base = np.exp(
            -bvals[None, :]
            * cos2
            * Dpar
        )  # (Nb, M)
    
        restricted = base * weighted  # (Nb, M)
    
        signal = np.sum(restricted * weights_bins[:, None], axis=0)  # (M,)
    
        return signal

    def Orientation(self,theta_f,phi_f,kappa):
        """
        Generate fibre orientations and their corresponding weights.

        Without dispersion, a single orientation defined by ``theta_f`` and
        ``phi_f`` is returned. With dispersion, a spherical discretization is
        constructed and weighted according to a Watson distribution centered
        on the specified fibre direction.

        Parameters
        ----------
        theta_f : float
            Central fibre polar angle in radians.

        phi_f : float
            Central fibre azimuthal angle in radians.

        kappa : float or None
            Watson concentration parameter. If ``None``, no orientation
            dispersion is applied.

        Returns
        -------
        V_bins : np.ndarray
            Array of orientation unit vectors with shape ``(N_bins, 3)``.

        weights_bins : np.ndarray
            Normalized weights associated with each orientation.

        Notes
        -----
        Larger ``kappa`` values produce stronger concentration around the
        central fibre direction.
        """
        st = np.sin(theta_f)
        if( kappa is None):
            V_bins = np.array([[
                st * np.cos(phi_f),
                st * np.sin(phi_f),
                np.cos(theta_f),
            ]])
            weights_bins = np.array([1.0])
            return V_bins,weights_bins
        else:
            V_unit = np.array([
                st * np.cos(phi_f),
                st * np.sin(phi_f),
                np.cos(theta_f),
            ])
        
            # spherical bins, shape (Nb, 3)
            u_edges = np.linspace(-1.0, 1.0, self.bins_theta + 1)
            theta_edges = np.arccos(u_edges)
            phi_edges = np.linspace(0.0, 2.0 * np.pi, self.bins_phi + 1)
        
            theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        
            Theta, Phi = np.meshgrid(theta_c, phi_c, indexing="ij")
            st = np.sin(Theta).ravel()
        
            V_bins = np.column_stack([
                st * np.cos(Phi).ravel(),
                st * np.sin(Phi).ravel(),
                np.cos(Theta).ravel(),
            ])
        
            # Watson weights, shape (Nb,)
            dots = V_bins @ V_unit
            weights_bins = np.exp(kappa * dots**2)
            weights_bins /= weights_bins.sum()
    
            return V_bins,weights_bins

    def SizeGen(self,size):
        """
        Generate compartment radii and their associated weights.

        The radius representation depends on the configured size model. A
        predefined discrete distribution is used for ``'Distribution'``, a
        single fixed radius is used for ``'Single'``, and an inferred size is
        converted into a Poisson-weighted radius distribution otherwise.

        Parameters
        ----------
        size : str or float
            Radius specification. ``'Distribution'`` selects a predefined
            distribution, ``'Single'`` selects a fixed radius, and a numerical
            value is interpreted as the characteristic inferred size.

        Returns
        -------
        R : np.ndarray
            Radius values used in the signal calculation.

        wR : np.ndarray
            Normalized weight associated with each radius.
        """
        if size == 'Distribution':
            R=np.array([1.5 ,2.5 ,3.5 ,4.5 ,5.5 ,6.5])*1e-3

            wR=np.array([1.826 ,9.2374 ,16.7562 ,22.986 ,18.525 ,16.8636])
            wR = wR/np.sum(wR)
            
        elif size == 'Single':
            R = np.array([3.5e-3])
            wR = np.array([1.0])
        else:
            # radius distribution
            R = np.linspace(1e-4, 1e-2, 100)
    
        
            lam_a = size * 1e4
            r_index = (R * 1e4).astype(int)
        
            wR = poisson.pmf(r_index, lam_a)
            wR /= wR.sum()          # (Nr,)
        return R,wR

#----------------Ball Compartment---------------------
class Balls(Compartment):
    """
    Restricted spherical diffusion compartment with optional size inference.

    The compartment models diffusion inside spherical restrictions. The
    spherical diffusivity is sampled directly, while the sphere radius can
    either be inferred, represented by a predefined distribution, or fixed to
    a single value.

    Parameters
    ----------
    gtab : object
        Gradient table containing gradient directions, b-values, and diffusion
        timing information.

    D_sph_min : float, optional
        Minimum spherical diffusivity used for sampling. Default is 0.

    D_sph_max : float, optional
        Maximum spherical diffusivity used for sampling. Default is 5e-3.

    Size : {'Infer', 'Distribution', 'Single'}, optional
        Strategy used for sphere-radius modelling. Default is ``'Infer'``.

    size_min : float, optional
        Minimum radius used when ``Size='Infer'``. Default is 1e-3.

    size_max : float, optional
        Maximum radius used when ``Size='Infer'``. Default is 20e-3.
    """
    def __init__(self,gtab,D_sph_min=0,D_sph_max=5e-3,Size='Infer',size_min=1e-3,size_max=20e-3):
        super().__init__(gtab)

        self.D_sph_range  = [D_sph_min,D_sph_max]
        self._check_range("D_sph", D_sph_min, D_sph_max)
        self.size  = Size

        allowed = {"Infer", "Distribution", "Single"}

        if Size not in allowed:
            raise ValueError(
                f"comp_type must be one of {allowed}, got '{Size}'"
            )

        if Size == 'Infer':
            if len(np.unique(self.gtab.big_delta))  < 2:
                print('*****WARNING: You are trying to fit size with only 1 diffusion time - proceed with caution!*****')
            self.size_range = [size_min,size_max]
            self._check_range("size", size_min, size_max)

        self.sph_bessel_roots = np.array(self.j1prime_zeros_spherical(20, x_max=10e6, step=0.01))
        
        self.build_parameter_list()
        
    def build_parameter_list(self):
        """
        Construct the parameter list used by the compartment.

        The spherical diffusivity is always included. The sphere radius is
        additionally included when ``Size='Infer'``.

        Returns
        -------
        None
            Parameter names are stored in ``self.parameter_names``.
        """
        self.parameter_names = ['D_sph']
        if(self.size == 'Infer'): self.parameter_names.append('size')


    def sample(self,N, rng = None):
        """
        Sample random parameter sets for the spherical compartment.

        Spherical diffusivity is sampled uniformly from its configured range.
        If sphere size is inferred, radius values are also sampled uniformly
        from the configured size range.

        Parameters
        ----------
        N : int
            Number of parameter sets to generate.

        rng : numpy.random.Generator or None, optional
            Random number generator used for sampling. If ``None``, a new
            NumPy random generator is created. Default is ``None``.

        Returns
        -------
        params : np.ndarray
            Sampled compartment parameters.

            When ``Size='Infer'``, the output has shape ``(N, 2)`` and contains
            ``[D_sph, size]``.

            Otherwise, the output contains only the sampled spherical
            diffusivities.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        D_sph_samples = rng.uniform(*self.D_sph_range, size=N)
        params = np.copy(D_sph_samples)
        if(self.size == 'Infer'):
            size_samples = rng.uniform(*self.size_range, size=N)
            params = np.column_stack([D_sph_samples, size_samples])
        return params
    
    def simulation(self, params):
        """
        Simulate signals for one or more spherical-compartment parameter sets.

        Restricted diffusion inside spheres is calculated using a truncated
        spherical-Bessel expansion and the diffusion timing information from
        the gradient table. Radius distributions are integrated using the
        weights returned by ``SizeGen``.

        Parameters
        ----------
        params : array-like
            Compartment parameter values.

            When ``Size='Infer'``, each parameter set contains
            ``[D_sph, size]``. Otherwise, only the spherical diffusivity is
            supplied and the configured size model is used.

        Returns
        -------
        restricted_sph : np.ndarray
            Simulated restricted spherical signals. The first dimension
            corresponds to parameter sets and the second to acquisitions.

        Notes
        -----
        The signal calculation uses the first 10 precomputed positive roots of
        the derivative of the spherical Bessel function ``j1``.
        """

        params = np.atleast_2d(np.asarray(params))
        
        if self.size == 'Infer':
            D_sph, size = params.T
        else:
            D_sph = params[0]
            size = self.size

        bvecs = np.asarray(self.gtab.bvecs, dtype=np.float64)
        bvals = np.asarray(self.gtab.bvals, dtype=np.float64)
        delta = np.asarray(self.gtab.small_delta, dtype=np.float64)
        Delta = np.asarray(self.gtab.big_delta, dtype=np.float64)

            
        Rs,wRs = self.SizeGen(size,len(D_sph))
        R2s = Rs**2

        m = 10
        brs  = self.sph_bessel_roots[:m]
        br2s = brs**2
        br6s = brs**6
    
        D_sph_ = D_sph[:, None, None, None]      # (N, 1, 1, 1)
        R2s_   = R2s[None, :, None, None]        # (1, Nr, 1, 1)
        br2s_  = br2s[None, None, :, None]       # (1, 1, m, 1)
        br6s_  = br6s[None, None, :, None]       # (1, 1, m, 1)
        delta_ = delta[None, None, None, :]      # (1, 1, 1, M)
        Delta_ = Delta[None, None, None, :]      # (1, 1, 1, M)
        
        num_s = (
            2 * D_sph_ * br2s_ * delta_ / R2s_ - 2
            + 2 * np.exp(-D_sph_ * br2s_ * delta_ / R2s_)
            + 2 * np.exp(-D_sph_ * br2s_ * Delta_ / R2s_)
            - np.exp(-D_sph_ * br2s_ * (Delta_ - delta_) / R2s_)
            - np.exp(-D_sph_ * br2s_ * (Delta_ + delta_) / R2s_)
        )
        den_s = (D_sph_**2 * br6s_ * (br2s_ - 2.0)/ R2s_**3)
        sumRs = np.sum(num_s / den_s, axis=2)
    
        x_s = -2.0 * bvals[None, :] / ((Delta[None, :] - delta[None, :] / 3.0) * delta[None, :]**2)
        Y = np.exp(x_s[:, None, :] * sumRs)
        restricted_sph = np.sum(Y * wRs[:, :, None], axis=1)

        return restricted_sph
    
    def SizeGen(self, size,N):
        """
        Generate sphere radii and their corresponding weights.

        The returned radius representation depends on the configured size
        model. A predefined distribution is used for ``'Distribution'``, a
        single fixed radius is used for ``'Single'``, and numerical inferred
        sizes are converted into Poisson-weighted radius distributions.

        Parameters
        ----------
        size : str or array-like
            Radius specification. For inferred sizes, this contains one radius
            estimate per parameter set.

        N : int
            Number of parameter sets for which radius weights are required.

        Returns
        -------
        R : np.ndarray
            Radius values used in the signal calculation.

        wR : np.ndarray
            Radius weights with shape ``(N, n_radii)``.

        Notes
        -----
        For inferred sizes, a radius grid between ``1e-4`` and ``3e-2`` is
        used and each input size determines a Poisson distribution over that
        grid.
        """
        if self.size == "Distribution":
            R = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]) * 1e-3
    
            wR = np.array([1.826, 9.2374, 16.7562, 22.986, 18.525, 16.8636])
            wR /= np.sum(wR)

            wR = np.tile(wR[None, :], (N, 1))
        elif self.size == "Single":
            R = np.array([3.5e-3])
            wR = np.ones((N, 1))
            
        else:
            size = np.asarray(size)
    
            R = np.linspace(1e-4, 3e-2, 100)
            r_index = (R * 1e4).astype(int)
    
            lam = size[:, None] * 1e4
            wR = poisson.pmf(r_index[None, :], lam)
            wR /= wR.sum(axis=1, keepdims=True)
    
        return R, wR

    def j1prime_zeros_spherical(self,n, x_max=100, step=0.1):
        """
        Find the first n positive roots of the derivative of spherical j1(x).
        
        Parameters
        ----------
        n     : int
            Number of roots to find
        x_max : float
            Maximum x to search
        step  : float
            Step size for scanning sign changes
        
        Returns
        -------
        zeros : list of float
            List of the first n roots (x > 0) of j1'(x).
        """
        zeros = []
        x_vals = np.arange(0.01, x_max, step)  # Start from small positive value to avoid division by zero
        
        f_prev = self.spherical_jn_derivative(1, x_vals[0])
        for i in range(1, len(x_vals)):
            f_curr = self.spherical_jn_derivative(1, x_vals[i])
            if f_prev * f_curr < 0:
                root = bisect(lambda x: self.spherical_jn_derivative(1, x), x_vals[i-1], x_vals[i])
                zeros.append(root)
                if len(zeros) == n:
                    break
            f_prev = f_curr
        
        return zeros
    def spherical_jn_derivative(self,n, x):
        """
        Derivative of the spherical Bessel function j_n(x).
        scipy provides this directly with derivative=True.
        """
        return spherical_jn(n, x, derivative=True)

#----------------Free Water Compartment---------------------
class FreeWater(Compartment):
    """
    Free-water diffusion compartment.

    This compartment models isotropic Gaussian diffusion with a fixed
    diffusivity. It contains no inferred model parameters; all simulated
    signals are determined solely by the acquisition b-values and the fixed
    free-water diffusivity.

    Parameters
    ----------
    gtab : object
        Gradient table containing the acquisition b-values.

    D_FW : float, optional
        Fixed free-water diffusivity. Default is 3e-3.
    """
    def __init__(self, gtab, D_FW=3e-3):
        super().__init__(gtab)
        self.D_FW = D_FW
        self.build_parameter_list()

    def build_parameter_list(self):
        """
        Construct the parameter list for the free-water compartment.

        Free-water diffusivity is fixed rather than inferred, so this
        compartment does not contribute any parameters to the model parameter
        vector.

        Returns
        -------
        None
            An empty list is stored in ``self.parameter_names``.
        """
        self.parameter_names = []

    def sample(self, N,rng = None):
        """
        Generate parameter arrays for the free-water compartment.

        Because the compartment contains no inferred parameters, the returned
        array has zero columns.

        Parameters
        ----------
        N : int
            Number of parameter sets to generate.

        rng : numpy.random.Generator or None, optional
            Random number generator. This argument is accepted for consistency
            with other compartment classes but is not used.

        Returns
        -------
        params : np.ndarray
            Empty parameter array with shape ``(N, 0)``.
        """
        return np.empty((N, 0))

    def simulation(self, params):
        """
        Simulate free-water diffusion signals.

        The signal is calculated using a mono-exponential isotropic diffusion
        model with the fixed diffusivity ``D_FW``.

        Parameters
        ----------
        params : array-like
            Parameter array used only to determine the number of simulations.
            Since this compartment has no inferred parameters, the final
            dimension may be empty.

        Returns
        -------
        signals : np.ndarray
            Simulated free-water signals with shape
            ``(N, n_measurements)``, where ``N`` is the number of supplied
            parameter sets.

        Notes
        -----
        The signal for each acquisition is calculated as

        ``exp(-b * D_FW)``,

        where ``b`` is the acquisition b-value and ``D_FW`` is the fixed
        free-water diffusivity.
        """
        params = np.atleast_2d(params)
        N = params.shape[0]

        bvals = np.asarray(self.gtab.bvals, dtype=np.float64).reshape(-1)

        signal = np.exp(-bvals * self.D_FW)

        return np.tile(signal[None, :], (N, 1))


#--------------------Model----------------------------------
class Model:
    """
    Multi-compartment diffusion model.

    The model combines one or more diffusion compartments into a weighted
    signal mixture. Compartment fractions are represented explicitly and
    compartment-specific parameters are appended to a global parameter list.

    The model can generate random parameter sets, simulate corresponding
    signals, optionally add Rician noise, normalize signals using b=0
    measurements, and save generated datasets.

    Parameters
    ----------
    gtab : object
        Gradient table containing acquisition information such as b-values,
        gradient directions, and diffusion timing.

    Attributes
    ----------
    gtab : object
        Gradient table used by all compartments.

    compartments : dict
        Dictionary containing the compartments currently included in the
        model.

    parameter_list : list of str
        Names of all compartment fractions and compartment-specific
        parameters.

    snr : float or None
        Default signal-to-noise ratio used when simulating noisy data.
    """
    def __init__(self, gtab,Normalize = False, S0_min = 0, S0_max = 2_000):
        self.gtab = gtab
        self.compartments = {}
        self.parameter_list = []
        self.snr = None
        self.Unnormalize = False
        if(not Normalize):
            self.Unnormalize = True
            self.S0_range = [S0_min, S0_max]


    def _set_default_SNR(self,SNR):
        """
        Set the default signal-to-noise ratio used during simulation.

        Parameters
        ----------
        SNR : float or None
            Default signal-to-noise ratio. If ``None``, noise is disabled
            unless a custom SNR is supplied during simulation.

        Returns
        -------
        None
            The SNR value is stored in ``self.snr``.
        """
        self.snr = SNR

    def _add_compartment(self, comp_type, name="", **kwargs):
        """
        Add a diffusion compartment to the model.

        The requested compartment class is initialized using the model's
        gradient table and any additional keyword arguments. The global
        parameter list is then rebuilt to include compartment fractions and
        all compartment-specific parameters.

        Parameters
        ----------
        comp_type : {'DTI', 'Sticks', 'Balls', 'FreeWater'}
            Type of compartment to add.

        name : str, optional
            Name assigned to the compartment. This name is used as a prefix
            for all associated parameters. Default is an empty string.

        **kwargs
            Additional keyword arguments passed to the constructor of the
            selected compartment class.

        Returns
        -------
        None
            The compartment is added to ``self.compartments`` and
            ``self.parameter_list`` is updated.

        Raises
        ------
        ValueError
            If ``comp_type`` is not one of the supported compartment types.
        """
        compartment_classes = {
            "DTI": DTI,
            "Sticks": Sticks,
            "Balls": Balls,
            "FreeWater":FreeWater
        }
    
        if comp_type not in compartment_classes:
            raise ValueError(
                f"comp_type must be one of {list(compartment_classes)}, got '{comp_type}'"
            )
    
        compartment = compartment_classes[comp_type](self.gtab, **kwargs)
        self.compartments[name] = compartment
        self.parameter_list = [f"{name}_f" for name, compartment in self.compartments.items()] + [
                    f"{name}_{p}"
                    for name, compartment in self.compartments.items()
                    for p in compartment.parameter_names
                ]
        if(self.Unnormalize):
            self.parameter_list += ["S0"]
    @property
    def _get_compartments(self):
        """
        Return the names and classes of all compartments in the model.

        Returns
        -------
        compartments : list of tuple
            List of ``(name, class_name)`` pairs for all currently registered
            compartments.
        """
        return [
            (name, type(compartment).__name__)
            for name, compartment in self.compartments.items()
        ]

    @property
    def _get_parameter_names(self):
        """
        Return the complete model parameter list.

        Returns
        -------
        parameter_list : list of str
            Names of all compartment fractions and compartment-specific
            parameters.
        """
        return self.parameter_list

    def _delete_compartment(self, name):
        """
        Remove a compartment from the model.

        After deletion, the global parameter list is rebuilt from the
        remaining compartments.

        Parameters
        ----------
        name : str
            Name of the compartment to remove.

        Returns
        -------
        None
            The selected compartment is removed and the parameter list is
            updated.

        Raises
        ------
        ValueError
            If no compartment with the requested name exists.
        """
        if name not in self.compartments:
            raise ValueError(f"No compartment named '{name}'")
        
        del self.compartments[name]
        self.parameter_list = [f"{name}_f" for name, compartment in self.compartments.items()] + [
            f"{name}_{p}"
            for name, compartment in self.compartments.items()
            for p in compartment.parameter_names
        ]

    def _get_parameter_index(self, name):
        """
        Find the index of a parameter in the model parameter list.

        Parameters
        ----------
        name : str
            Exact name of the parameter to locate in ``self.parameter_list``.

        Returns
        -------
        index : int
            Index of the parameter in ``self.parameter_list``.

        Raises
        ------
        ValueError
            If ``name`` is not found in ``self.parameter_list`` or if it appears
            more than once.

        Notes
        -----
        Parameter names are matched using exact string equality.
        """
        matches = np.flatnonzero([name == p for p in self.parameter_list])

        if len(matches) == 0:
            raise ValueError(f"Parameter {name} was not found.")

        if len(matches) > 1:
            raise ValueError(f"Parameter {name} appears multiple times.")

        return matches[0]

    def simulation(self,parameters,S0 = None,parameter_list = None,custom_snr = 0,rng = None):
        """
        Simulate signals from supplied multi-compartment model parameters.

        Each compartment signal is simulated independently and weighted by its
        corresponding compartment fraction. The weighted signals are summed
        to produce the final model signal.

        Optional Rician noise and b=0 normalization are applied according to
        the selected signal-to-noise ratio.

        Parameters
        ----------
        parameters : array-like
            Model parameters with shape ``(N, n_parameters)``.

        parameter_list : sequence of str or None, optional
            Names corresponding to the columns of ``parameters``. If ``None``,
            ``self.parameter_list`` is used. Default is ``None``.

        custom_snr : float or None, optional
            Signal-to-noise ratio used for this simulation.

            If set to 0, ``self.snr`` is used. If set to ``None``, no noise or
            normalization is applied. Default is 0.

        rng : numpy.random.Generator or None, optional
            Random number generator used when adding noise. If ``None``, a new
            NumPy random generator is created. Default is ``None``.

        Returns
        -------
        S : np.ndarray
            Simulated signals with shape ``(N, n_measurements)``.

        Notes
        -----
        The total signal is formed as the sum of the compartment signals
        weighted by their corresponding fractions.

        If noise is enabled, Rician noise is added using ``_add_noise`` and
        the resulting signals are normalized using ``_normalize``.
        """

        if rng is None: rng = np.random.default_rng()

        N = parameters.shape[0]
        n_meas = len(self.gtab.bvals)
        S = np.zeros((N, n_meas))

        for key, value in self.compartments.items():
            rel_pars = [
                self._get_parameter_index(f"{key}_{name}")
                for name in value.parameter_names
            ]
            
            rel_frac = self._get_parameter_index(f"{key}_f")
            S += parameters[:,rel_frac,None]*value.simulation(parameters[:,rel_pars])

        snr = self.snr if custom_snr == 0 else custom_snr


        if(self.Unnormalise):
            if S0 == None:
                print("You didn't provide an S0 set, will set it to 1")
                S0 = np.ones(parameters.shape[0])
            S = S0[:,None] * S

        if snr is not None:
            S = self._add_noise(S,S0, snr,rng)
        
        if not self.Unnormalise: S = self._normalize(S)

        return S

    def sample_and_simulation(self,N,custom_snr=0,parallel=False,save = False,filename = None,rng = None, n_jobs = -1):
        """
        Sample random model parameters and simulate their corresponding signals.

        Compartment fractions are sampled from a Dirichlet distribution, while
        compartment-specific parameters are sampled independently by each
        compartment. The corresponding compartment signals are weighted by
        their fractions and summed.

        Simulation may be performed sequentially or in parallel. Optional
        Rician noise, b=0 normalization, and HDF5 saving are supported.

        Parameters
        ----------
        N : int
            Number of parameter sets and signals to generate.

        custom_snr : float or None, optional
            Signal-to-noise ratio used for generated data.

            If set to 0, ``self.snr`` is used. If set to ``None``, the raw
            noiseless signals are returned. Default is 0.

        parallel : bool, optional
            If ``True``, compartment simulations are performed using each
            compartment's parallel simulation method. Default is ``False``.

        save : bool, optional
            If ``True``, save the generated parameters and signals to an HDF5
            file. Default is ``False``.

        filename : str or None, optional
            Filename used when saving. If ``None``, the saving function
            generates a filename automatically. Default is ``None``.

        rng : numpy.random.Generator or None, optional
            Random number generator used for fraction sampling, compartment
            sampling, and noise generation. If ``None``, a new NumPy random
            generator is created. Default is ``None``.

        n_jobs : int, optional
            Number of parallel workers used when ``parallel=True``.
            A value of -1 uses all available workers. Default is -1.

        Returns
        -------
        Params : np.ndarray
            Sampled model parameters, including all compartment fractions and
            compartment-specific parameters.

        S : np.ndarray
            Simulated signals after optional noise addition and normalization.

        Raises
        ------
        ValueError
            If the model contains no compartments.

        Notes
        -----
        Compartment fractions are sampled jointly from a symmetric Dirichlet
        distribution, ensuring that they are non-negative and sum to one.

        If ``save=True``, both the processed signal and the raw noiseless
        signal are passed to ``Helpers.save_h5``.
        """

        if rng is None:
            rng = np.random.default_rng()
        if len(self.compartments) == 0:
            raise ValueError(f"Need at least one compartment!")
    
        n_meas = len(self.gtab.bvals)
        n_comp = len(self.compartments)
        
        S_raw = np.zeros((N, n_meas))
        fracs = rng.dirichlet(alpha=np.ones(n_comp), size=N)
        all_params = [fracs]
        pbar = tqdm.tqdm(enumerate(self.compartments.items()),position=0)

        if parallel:
            for i, (name, compartment) in pbar:
                pbar.set_description(f"Simulating {name}")
                params, c_sim = compartment.sample_and_simulate_parallel(N,rng=rng,n_jobs=n_jobs)
                all_params.append(params)
                S_raw += fracs[:, i, None] * c_sim
        else:
            for i, (name, compartment) in pbar:
                pbar.set_description(f"Simulating {name}")
                params, c_sim = compartment.sample_and_simulate(N,rng=rng)
                all_params.append(params)
                S_raw += fracs[:, i, None] * c_sim

        Params = np.hstack(all_params)
    
        snr = self.snr if custom_snr == 0 else custom_snr

        if self.Unnormalize: 
            S0Rand = self._sample_S0(N,rng)
            Params = np.hstack([Params, S0Rand[:, None]])
        else:
            S0Rand = np.ones(N)

        S = S0Rand[:,None] * S_raw

        if snr is None: 
            pass
        else:
            S = self._add_noise(S,S0Rand, snr,rng)

        if(not self.Unnormalize):
            S = self._normalize(S)
        
        if save: Helpers.save_h5(filename, Params, S, S_raw, snr,self.parameter_list,self.compartments)

        return Params, S


    def _sample_S0(self,N,rng):

        return rng.uniform(*self.S0_range, size=N)

    def _add_noise(self, S,S0, snr,rng):
        """
        Add Rician noise to simulated magnitude signals.

        Two independent Gaussian noise components are generated and combined
        with the noiseless signal to produce Rician-distributed magnitude data.

        Parameters
        ----------
        S : np.ndarray
            Noiseless signal array.

        snr : float
            Signal-to-noise ratio. The Gaussian noise standard deviation is
            calculated as ``1 / snr``.

        rng : numpy.random.Generator or None
            Random number generator used for noise generation. If ``None``, a
            new NumPy random generator is created.

        Returns
        -------
        noisy_signal : np.ndarray
            Signal array after addition of Rician noise.
        """
        if rng is None:
            rng = np.random.default_rng()
        sigma = S0[:,None] / snr
    
        noise1 = rng.normal(0, sigma, size=S.shape)
        noise2 = rng.normal(0, sigma, size=S.shape)
    
        return np.sqrt((S + noise1)**2 + noise2**2)

    def _normalize(self, S):
        """
        Normalize signals by their mean b=0 signal.

        For each simulated signal, all measurements with ``b=0`` are averaged
        to obtain a baseline signal ``S0``. Every measurement is then divided
        by this baseline.

        Parameters
        ----------
        S : np.ndarray
            Signal array with shape ``(N, n_measurements)``.

        Returns
        -------
        normalized_signal : np.ndarray
            Signal array normalized by the mean b=0 signal for each sample.

        Raises
        ------
        ValueError
            If the gradient table does not contain any b=0 measurements.
        """
        b0_mask = self.gtab.bvals == 0

        if not np.any(b0_mask):
            raise ValueError("Cannot normalize: no b=0 measurements found.")

        S0 = S[:, b0_mask].mean(axis=1, keepdims=True)

        return S / S0