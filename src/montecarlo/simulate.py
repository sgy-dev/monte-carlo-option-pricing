# Author: Shaun Geaney, Phd
# Date: 17-Apr-2026

import numpy as np
import numpy.random as npr

from dataclasses import dataclass


@dataclass
class EuropeanOption:
    """Class to represent a European call option and its parameters.
     Attributes:
        S0: Initial stock price [USD]
        K: Strike price [USD]
        T: Time to maturity [Y]
        r: Constant, risk-free rate [%] (Continuously compounded i.e., no dividends.)
        sigma: Volatility of the underlying [%]
    """
    S0: float = 100.0
    K: float = 105.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2

    @property
    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)
    

@dataclass
class MonteCarloSimulator:
    """Class to perform Monte Carlo simulations for European option pricing.
    Attributes:
        option: An instance of the EuropeanOption class containing option parameters.
        num_t_steps: Number of time steps to simulate (default: 250 for average number of trading days in a year).
        num_paths: Number of Monte Carlo paths to simulate (default: 1,000).
    """

    option: EuropeanOption 
    num_t_steps: int = 250
    num_paths: int = 1_000

    def __post_init__(self):
        self.dt: np.ndarray = self.option.T / self.num_t_steps
        self.t: np.ndarray = np.linspace(0, self.option.T, self.num_t_steps + 1)
        self.results: np.ndarray = np.empty((self.num_paths, self.num_t_steps + 1), dtype=np.float64)

    def calculate_St(self) -> None:
        """Calculate the stock price paths using the geometric Brownian motion model using 'num_t_steps' number of time steps.
        The stock price at time t is given by:
            S(t) = S0 exp((r - ½σ²)t + σWt)
        where Wt is a Wiener process (Brownian motion) with mean 0 and variance dt.
        We simulate Wt using the cumulative sum of normally distributed random variables.
        Note: This method should be called before running simulations to ensure that the stock price paths are calculated.
        """
        # Normal distribution, N(mean=0, var=dt)
        Wt: np.ndarray = np.cumsum(npr.normal(0, np.sqrt(self.dt), self.num_t_steps))
        return self.option.S0 * np.exp((self.option.r - 0.5 * self.option.sigma ** 2) * self.t + self.option.sigma * Wt) 
    
    def run_simulations(self) -> None:
        raise(NotImplementedError)
    
    def calculate_avg(self) -> None:
        raise(NotImplementedError)
    
    def plot_simulations(self) -> None:
        raise(NotImplementedError)
    
    