"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
from .stationarity import Stationary

class Surrogate(Stationary):
    """
    A class to represent a surrogate process on a graph.
    This class provides methods to generate random surrogates of a given signal
    using various randomization techniques, including direct and non-direct methods.
    It inherits from the Stationary class to utilize its methods for checking stationarity
    and estimating covariance and auto-correlation of graph samples.
    """

    def __init__(self, graph, params=None):
        super().__init__(graph, params=params)
        if self.graph.operator is None:
            print("Warning: No operator set for the graph. ")
            print("Defaulting to adjacency operator.")
            self.graph.set_operator(name='adjacency')

    def randomizer_phase(self, N:int, seed:int, 
                        conj:bool=True, onlysign:bool=False):
        """
        Compute randomizing vector

        Parameters:
        -----------
        N : int
            The size of the randomizing vector.
        seed : int
            The seed to use for the random number generator.
        conj : bool, optional
            Whether to use the conjugate of the random shift, by default False.
        onlysign : bool, optional
            Whether to only randomize the sign, by default False.

        Returns:
        --------
        randomizer_vector : np.ndarray
            The computed randomizing vector.
        """
        tasks = self.graph.operator.eigvalues_pairs()
        np.random.seed(seed)
        randomizer_vector = np.zeros((N), dtype=complex)
        for t in tasks:
            if len(t) == 1:
                # Flip Sign with uniform probability
                randomval = np.random.random()
                mask = -1 * (randomval > 0.5) + 1 * (randomval <= 0.5)
                randomizer_vector[t[0]] = mask
            elif len(t) == 2:
                # Rotate complex value by uniform proba angle
                if onlysign:
                    random_shift1 = np.random.random()
                    random_shift2 = np.random.random()
                    s1 = -1 * (random_shift1 > 0.5) + 1 * (random_shift1 <= 0.5)
                    if conj:
                        s2 = s1
                    else:
                        s2 = -1 * (random_shift2 > 0.5) + 1 * (random_shift2 <= 0.5)
                else:
                    random_shift1 = np.random.random() * 2 * np.pi
                    random_shift2 = np.random.random() * 2 * np.pi
                    s1 = np.exp(1j * random_shift1)
                    if conj:
                        s2 = np.exp(-1j * random_shift1)
                    else:
                        s2 = np.exp(1j * random_shift2)

                randomizer_vector[t[0]] = s1
                randomizer_vector[t[1]] = s2
            else:
                for tidx in range(len(t)):
                    random_shift = np.random.random() * 2 * np.pi
                    randomizer_vector[t[tidx]] = np.exp(1j * random_shift)
                
        randomizer_vector = np.diag(randomizer_vector)
        return randomizer_vector

    def phase_randomize(self, signal:np.ndarray, seed:int=99):
        """
        Randomizes the given signal using the provided transformation matrices and a random seed.

        Parameters:
        ----------
        signal (numpy.ndarray): The input signal to be randomized.
        seed (int, optional): The seed for the random number generator. Default is 99.

        Returns:
        --------
        numpy.ndarray: The randomized signal.
        """
        N = signal.shape[0]
        complex_randomizer = self.randomizer_phase(N, seed=seed, conj=True)
        gf_coef = self.graph.operator.GFT(signal)
        randomized = self.graph.operator.inverseGFT(complex_randomizer @ gf_coef)
        return randomized

    def naive_random_surrogate(self, signal:np.ndarray, nrands:int=99, seed:int=99):
        """
        Generate nrands number of naive random surrogates for the input array arr.

        The surrogates are generated by randomly permuting the input array arr.

        Parameters
        ----------
        signal : np.ndarray
            Input signal to generate surrogates for
        nrands : int, optional
            Number of random surrogates to generate. Default is 99. 
        seed : int, optional 
            Random seed for reproducibility. Default is 99.

        Returns
        -------
        ret : np.ndarray
            Array of shape (nrands, len(arr)) containing the random surrogates.
        """
        np.random.seed(seed)

        ret = np.zeros((nrands, len(signal)))
        for i in range(nrands):
            ret[i] = signal[np.random.permutation(len(signal))]
        return ret

    def undirected_random_surrogate(self, signal: np.ndarray, 
                                    nrands:int=99, rseed:int=99):
        """
        Undirected informed generation of surrogate signals
        #TODO use phase randomizer instead of randomizing the sign here
        Parameters
        ----------
        signal : np.ndarray
            The input signal array
        U : np.ndarray 
            The graph Fourier basis
        Uinv : np.ndarray
            The inverse graph Fourier basis  
        nrands : int, optional
            The number of random surrogates to generate (default 99)
        rseed : int, optional 
            The random seed (default 99)

        Returns
        -------
        ret : list
            A list containing the randomized surrogate signals
        """
        np.random.seed(rseed)
        ssignal = self.graph.operator.GFT(signal)

        ret = []
        for _ in tqdm(range(nrands), disable=True):
            # Initialize the randomizer
            R = np.diag(np.sign(np.random.random(self.graph.N) - 0.5))

            rand = R @ ssignal
            randomized = self.graph.operator.inverseGFT(rand)
            ret.append(randomized)

        return ret
    
    def directed_random_surrogate(self, signal:np.ndarray, nrands:int, 
                                  seed:int=99, normalize:bool=False):
        """
        Generate surrogate data by randomizing the given signal using a direct method.

        Parameters
        ----------
        signal : np.ndarray
            The input signal to be randomized.
        nrands : int
            The number of random surrogates to generate.
        seed : int, optional
            The seed for the random number generator (default is 99).
        normalize : bool, optional
            If True, normalize the surrogates to have the same norm as the input signal (default is False).

        Returns
        -------
        surrogates : np.ndarray
            An array of generated surrogate signals.
        """
        np.random.seed(seed)
        seeds = (10000 * np.random.random(nrands)).astype(int)
        surrogates = np.array([self.phase_randomize(signal, seed=seed_idx) for seed_idx in seeds]).real
        if normalize:
            surrogates = np.linalg.norm(signal) * surrogates / np.linalg.norm(surrogates, axis=1)[:, None]
        return surrogates