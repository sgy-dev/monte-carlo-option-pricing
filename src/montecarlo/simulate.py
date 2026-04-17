# Author: Shaun Geaney, Phd
# Date: 17-Apr-2026

import numpy as np

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
    
    