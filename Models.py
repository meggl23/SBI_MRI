from abc import ABC, abstractmethod
from joblib import Parallel, delayed

import numpy as np
from scipy.special import j0, jv
from scipy.special import spherical_jn
from scipy.optimize import bisect
from scipy.stats import poisson

#----------------Generic Compartment---------------------
class Compartment(ABC):
    def __init__(self,gtab):
        self.gtab = gtab
    
    @abstractmethod
    def simulation(self):
        pass

    @abstractmethod
    def sample(self):
        pass
        
    def sample_and_simulate(self, N=1):
        params = self.sample(N)
        signals = self.simulation(params)
        return params, signals
        
    @abstractmethod
    def build_parameter_list(self):
        pass
        
    def sample_and_simulate_parallel(self, N=1, n_jobs=-1, chunk_size=1000):
        params = self.sample(N)
    
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
    def __init__(self,gtab,MD_min = 0, MD_max = 0.005, FA_min = 0, FA_max = 0.999):
        super().__init__(gtab)
        self.MD_range = [MD_min,MD_max]
        self.FA_range = [FA_min,FA_max]
        self.DTI_gen_max_tries = 10000

        self.build_parameter_list()
    
    def simulation(self, tensors):
        tensors = np.asarray([self.flat_to_tens(t) for t in tensors])
    
        if tensors.shape[1:] != (3, 3):
            raise ValueError("tensors must have shape (N, 3, 3)")
    
        gradients = self.gtab.bvecs.reshape(-1, 3)
        bvals = self.gtab.bvals.reshape(-1)
    
        adc = np.einsum("ij,njk,ik->ni", gradients, tensors, gradients)
    
        return np.exp(-bvals[None, :] * adc)
        
    def sample(self, N=1):
        MD_samples = np.random.uniform(*self.MD_range, size=N)
        FA_samples = np.random.uniform(*self.FA_range, size=N)
    
        tensors = np.array([
            self.tens_to_flat(self.random_diffusion_tensor(md, fa))
            for md, fa in zip(MD_samples, FA_samples)
        ])
    
        return tensors

    def build_parameter_list(self):
        self.parameter_names = ['D_xx','D_xy','D_yy','D_xz','D_yz','D_zz']

    def flat_to_tens(self, flat_tensor):
        xx, xy, yy, xz, yz, zz = flat_tensor
    
        return np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ])
    
    def tens_to_flat(self, tensor):
        return np.array([
            tensor[0, 0],
            tensor[0, 1],
            tensor[1, 1],
            tensor[0, 2],
            tensor[1, 2],
            tensor[2, 2],
        ])

    def random_diffusion_tensor(self,MD, FA):
        if MD <= 0:
            raise ValueError("MD must be positive")
    
        if not (0 <= FA < 1):
            raise ValueError("FA must satisfy 0 <= FA < 1")
    
        X = (3 * FA**2 * MD**2) / (1.5 - FA**2)
    
        for _ in range(self.DTI_gen_max_tries):
            theta = np.random.uniform(0, 2 * np.pi)
    
            a = np.sqrt(X / 6) * np.cos(theta)
            b = np.sqrt(X / 2) * np.sin(theta)
    
            l1 = MD + a + b
            l2 = MD + a - b
            l3 = MD - 2 * a
    
            lambdas = np.array([l1, l2, l3])
    
            if np.all(lambdas > 0):
                Q = self.random_SO3()
                return Q @ np.diag(lambdas) @ Q.T
    
        raise RuntimeError("Could not generate positive eigenvalues")
    
    def random_SO3(self):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
    
        # make Q uniformly distributed over SO(3)
        Q = Q @ np.diag(np.sign(np.diag(R)))
    
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
    
        return Q

    def MD_FA(self,tensor, clip_negative=False, eps=1e-5):
        evals = np.linalg.eigvalsh(tensor)
    
        if clip_negative:
            evals = np.maximum(evals, eps)
    
        MD = evals.mean()
    
        denom = np.sum(evals**2)
        if denom == 0:
            return MD, 0.0
    
        FA = np.sqrt(
            3 * np.sum((evals - MD)**2)
            / (2 * denom)
        )
    
        return [MD, FA]

#----------------Stick Compartment---------------------
class Sticks(Compartment):
    def __init__(self,gtab, D_par_min = 0, D_par_max = 5e-3, D_perp_min = 0,D_perp_max = 5e-3,
                Dispersion=False,Size = 'Distribution'):
        super().__init__(gtab)

        self.D_par_range  = [D_par_min,D_par_max]
        self.D_perp_range = [D_perp_min,D_perp_max]
        
        self.dispersion = False
        self.size  = Size

        allowed = {"Infer", "Distribution", "Single"}

        if Size not in allowed:
            raise ValueError(
                f"comp_type must be one of {allowed}, got '{Size}'"
            )

        if Dispersion:
            self.kappa_range = [0,10]
            self.dispersion  = True
        if Size == 'Infer':
            if len(np.unique(self.gtab.small_delta))  < 2:
                print('*****WARNING: You are trying to fit size with only 1 diffusion time - proceed with caution!*****')
            size_min = 1e-4
            size_max = 0.005
            self.size_range = [size_min,size_max]
            
        n_roots = 15
        self.Bessel_roots = np.array(self.j1prime_zeros(n_roots, x_max=10e6, step=0.01))

        self.bins_theta=10
        self.bins_phi=20

        self.build_parameter_list()
        
    def build_parameter_list(self):
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

    def sample(self,N):
        
        D_par_samples = np.random.uniform(*self.D_par_range, size=N)
        D_perp_samples = np.random.uniform(*self.D_perp_range, size=N)


        V = np.random.randn(N, 3)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        Angs = np.array([self.SpherAng(v) for v in V])

        params = np.column_stack([Angs,D_par_samples,D_perp_samples])
        
        if(self.dispersion):
            kappa_samples = np.random.uniform(*self.kappa_range, size=N)
            params = np.column_stack([params, kappa_samples])

        if(self.size == 'Infer'):
            size_samples = np.random.uniform(*self.size_range, size=N)
            params = np.column_stack([params, size_samples])
        return params

    def SpherAng(self,vector):
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
        params = np.asarray(params)
    
        if params.ndim == 1:
            return self.simulate_one(params)[None, :]
    
        return np.vstack([self.simulate_one(p) for p in params])
        
    def simulate_one(self, params):
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
    def __init__(self,gtab,D_sph_min=0,D_sph_max=5e-3,Size='Infer'):
        super().__init__(gtab)

        self.D_sph_range  = [D_sph_min,D_sph_max]
        self.size  = Size

        allowed = {"Infer", "Distribution", "Single"}

        if Size not in allowed:
            raise ValueError(
                f"comp_type must be one of {allowed}, got '{Size}'"
            )

        if Size == 'Infer':
            if len(np.unique(self.gtab.small_delta))  < 2:
                print('*****WARNING: You are trying to fit size with only 1 diffusion time - proceed with caution!*****')
            size_min = 1e-3
            size_max = 20e-3
            self.size_range = [size_min,size_max]

        self.sph_bessel_roots = np.array(self.j1prime_zeros_spherical(20, x_max=10e6, step=0.01))
        
        self.build_parameter_list()
        
    def build_parameter_list(self):
        self.parameter_names = ['D_sph']
        if(self.size == 'Infer'): self.parameter_names.append('size')


    def sample(self,N):
        
        D_sph_samples = np.random.uniform(*self.D_sph_range, size=N)
        params = np.copy(D_sph_samples)
        if(self.size == 'Infer'):
            size_samples = np.random.uniform(*self.size_range, size=N)
            params = np.column_stack([D_sph_samples, size_samples])
        return params
    
    def simulation(self, params):
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
    def __init__(self, gtab, D_FW=3e-3):
        super().__init__(gtab)
        self.D_FW = D_FW
        self.build_parameter_list()

    def build_parameter_list(self):
        self.parameter_names = []

    def sample(self, N):
        return np.empty((N, 0))

    def simulation(self, params):
        params = np.atleast_2d(params)
        N = params.shape[0]

        bvals = np.asarray(self.gtab.bvals, dtype=np.float64).reshape(-1)

        signal = np.exp(-bvals * self.D_FW)

        return np.tile(signal[None, :], (N, 1))


#--------------------Model----------------------------------
class Model:
    def __init__(self, gtab):
        self.gtab = gtab
        self.compartments = {}
        self.parameter_list = []

    def _set_default_SNR(self,SNR):
        self.snr = SNR

    def _add_compartment(self, comp_type, name="", **kwargs):
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

    def _get_compartments(self):
        for name, compartment in self.compartments.items():
            print(f"{name}: {type(compartment).__name__}")
    def _get_parameter_names(self):
        print(self.parameter_list)

    def _delete_compartment(self, name):
    
        if name not in self.compartments:
            raise ValueError(f"No compartment named '{name}'")
        
        del self.compartments[name]
        self.parameter_list = [f"{name}_f" for name, compartment in self.compartments.items()] + [
            f"{name}_{p}"
            for name, compartment in self.compartments.items()
            for p in compartment.parameter_names
        ]
    def simulation(self,N,custom_snr=None,parallel=False):
        if len(self.compartments) == 0:
            raise ValueError(f"Need at least one compartment!")
    
        n_meas = len(self.gtab.bvals)
        n_comp = len(self.compartments)
        
        S = np.zeros((N, n_meas))
        fracs = np.random.dirichlet(alpha=np.ones(n_comp), size=N)
        all_params = [fracs]

        if(parallel):
            for i, compartment in enumerate(self.compartments.values()):
                params, c_sim = compartment.sample_and_simulate_parallel(N)
                all_params.append(params)
                S += fracs[:, i, None] * c_sim
        else:
            for i, compartment in enumerate(self.compartments.values()):
                params, c_sim = compartment.sample_and_simulate(N)
                all_params.append(params)
                S += fracs[:, i, None] * c_sim

        Params = np.hstack(all_params)
        snr = self.snr if custom_snr is None else custom_snr
        
        if snr is None:
            return Params, S
            
        return  Params, self._add_noise(S, snr)

    def _add_noise(self, S, snr):
        sigma = 1.0 / snr
    
        noise1 = np.random.normal(0, sigma, size=S.shape)
        noise2 = np.random.normal(0, sigma, size=S.shape)
    
        return np.sqrt((S + noise1)**2 + noise2**2)