# Monte Carlo Methods for Options Pricing

*This repo is primarily for personal pedagogical purposes.*

## Introduction

This project demonstrates how Monte Carlo simulation can be used to estimate the price of a European call option. A European call option gives the buyer the right, but not the obligation, to buy an underlying asset at a fixed strike price on a fixed future date. Monte Carlo methods are widely used in quantitative finance because they can model many possible future outcomes and estimate value when markets are uncertain.

## Monte Carlo Methods

The main idea behind Monte Carlo is to simulate (or sample from a probability distribution) the underlying asset price $S$, over time $t$, using a simple stochastic model called [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) that satisfies the stochastic differential equation

```math
dS_t = \mu S_t dt + \sigma S_t dW_t 
```

The price path is generated from the solution to the above equation which is

```math
S_t = S_0 \exp\Bigl((r - \tfrac{1}{2}\sigma^2) t + \sigma W_t\Bigr)
```

where

- $S_t$ is the simulated asset price at time $t$.
- $S_0$ is the initial asset price.
- $r$ is the continuously compounded risk-free rate
- $σ$ is the asset volatility
- $W_t$ is a random Brownian motion term (a cumulative sum of normal random increments)

By simulating many independent paths and observing the final asset prices, the method estimates the expected behavior of the option without solving the full analytic pricing formula.

**Note:** It is common in quantitative mathematics to denote an underlying stock value $S$, as a function of time $t$, as $S_t$ instead of $S(t)$. This avoids multiple messy brackets as there are many variables with time-dependence.

## Running the Code

If you wish to use or experiment with this project, `clone` it and install the package dependencies listed in the `pyproject.toml` file. Then run the notebook example from Python.

To test the simulator, run the `monte-carlo-example.ipynb` notebook. Or, you can create your own Python script and insert the following code snippet (feel free to change the example parameters suggested below):

```python
from montecarlo.simulate import EuropeanOption, MonteCarloSimulator

option = EuropeanOption(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2)
mcs = MonteCarloSimulator(option=option, num_t_steps=250, num_paths=1_000)
mcs.run_simulations()
mcs.plot_simulations()
```

## Output

The simulation process is depicted in the animation below. The simulation produces two plots:

1. All the simulated $S_t$ paths (for a given set of option parameters), along with a dashed line for the strike price $K$, and the average of all the paths in blue. The final value of all the averages paths gives the expected value of the underlying after time $T$, *i.e.,* $\mathbb{E}[S(t=T)]$.

2. The second plot is a horizontal histogram plot of the probability distribution of the underlying paths (this is a log-normal distribution). The green bins show path prices that finished above the strike price, and the red bins show prices below the strike. 

Whilst the code in the `main` branch produces a final, static image, there is an experimental `animate_simulations()` function on the `feat-animation-experiment` branch that is not thoroughly tested and can produce the animated .gif below.

![Monte Carlo Option Pricing Simulation](img/montecarlo.gif)

## References

- Plotting style inspired by [ebrahimpichka/mc-option-pricing](https://github.com/ebrahimpichka/mc-option-pricing).

- Theory and understanding were informed by [Quant Start](https://www.quantstart.com/articles/).