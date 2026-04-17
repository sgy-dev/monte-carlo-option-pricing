# Author: Shaun Geaney, Phd
# Date: 17-Apr-2026

import numpy as np
import numpy.random as npr

from dataclasses import dataclass


@dataclass
class EuropeanOption:
    S0: float = 100.0  # Initial stock price
    K: float = 105.0   # Strike price
    T: float = 1.0     # Time to maturity (in years)
    r: float = 0.05    # Risk-free interest rate
    sigma: float = 0.2 # Volatility of the underlying asset

    @property
    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)
    

@dataclass
class MonteCarloSimulator:

    option: EuropeanOption  # Option parameters
    num_t_steps: int = 250  # Avg number of trading days in a year
    num_paths: int = 1_000  # Number of Monte Carlo paths to simulate

    def __post_init__(self):
        self.dt: np.ndarray = self.option.T / self.num_t_steps
        self.t: np.ndarray = np.linspace(0, self.option.T, self.num_t_steps + 1)
        self.results: np.ndarray = np.empty((self.num_paths, self.num_t_steps + 1), dtype=np.float64)