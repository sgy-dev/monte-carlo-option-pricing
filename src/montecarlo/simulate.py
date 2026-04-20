# Author: Shaun Geaney, Phd
# Date: 17-Apr-2026

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm, trange

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
        self.t: np.ndarray = np.linspace(0, self.option.T, self.num_t_steps)
        self.results: np.ndarray = np.empty((self.num_paths, self.num_t_steps), dtype=np.float64)
        self.avg_path: np.ndarray = np.empty(self.num_t_steps, dtype=np.float64)

    def calculate_St(self) -> np.ndarray:
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
        """Run the Monte Carlo simulations to generate stock price paths.
        This method populates the 'results' attribute with the simulated stock price paths.
        Each row in 'results' corresponds to a single simulation path, and each column corresponds to a time step.
        """
        for i in trange(self.num_paths, desc="Running Monte Carlo simulations"):
            self.results[i, :] = self.calculate_St()
        self.calculate_avg()
    
    def calculate_avg(self) -> None:
        """Calculate the average stock price at each time step across all simulation paths.
        This method returns a 1D array containing the average stock price at each time step.
        """
        self.avg_path = np.mean(self.results, axis=0)
    
    def plot_simulations(self) -> None:
        """Plotting functions to show the simulated Monte Carlo paths and the resulting probability distribution."""

        # --- Create a figure with two subplots: one for the paths and one for the histogram ---
        fig, (ax_paths, ax_hist) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            sharey=True,
            gridspec_kw={
                "width_ratios": [3, 1],
                "wspace": 0.1}
            )

        # --- Plot the simulated paths ---
        ends_above_avg: bool = self.results[:, -1] > self.avg_path[-1]
        abv_clr, blw_clr = "green", "red"
        colours = np.where(ends_above_avg, abv_clr, blw_clr)

        # Plot each path with a color based on whether it ends above or below the average path.
        for i in trange(self.num_paths, desc="Plotting Monte Carlo paths"):
            ax_paths.plot(self.t, self.results[i, :], color=colours[i], lw=0.5, alpha=0.5)
        ax_paths.plot([], [] , color=abv_clr, label=r"Paths ends $\bf{above}$ avg. final price")
        ax_paths.plot([], [] , color=blw_clr, label=r"Paths ends $\bf{below}$ avg. final price")

        # Plot the average path if there are multiple paths.
        if self.num_paths > 1:    
            ax_paths.plot(self.t, self.avg_path, color="b", lw=2, label=f"Avg. path, $E[S(t=T)]$ = \\${self.avg_path[-1]:.2f}")
        
        # Plot the strike price as a horizontal dashed line.
        ax_paths.axhline(self.option.K, color="k", linestyle="--", lw=2)
        
        # labels, title, legend etc.
        ax_paths.set_title(
            f"Monte Carlo: Spot = \\${self.option.S0:.2f}, "
            + f"r = {self.option.r*100:.1f}%, "
            + f"σ = {self.option.sigma*100:.1f}%"
        )
        ax_paths.set_xlim(0, self.option.T)
        ax_paths.set_xlabel("Time, $t$ [Years]")
        ax_paths.set_ylabel("Stock Price, $S(t)$ [\\$]")
        ax_paths.legend(loc="upper left")

        # --- Plot the histogram of final stock prices ---
        num_bins = 200 if 200 < round(self.num_paths / 10) else round(self.num_paths / 10)
        ax_hist.hist(self.results[:, -1], bins=num_bins, orientation="horizontal", color="lightgray", density=True)
        ax_hist.axhline(self.option.K, color="k", linestyle="--", lw=2, label=f"Strike = \\${self.option.K:.2f}")
        
        # Color the histogram bins based on whether they are above or below the strike price.
        for patch in tqdm(ax_hist.patches, desc="Plotting histogram bins"):
            bin_center = patch.get_y() + patch.get_height() / 2
            patch.set_facecolor(abv_clr if bin_center > self.option.K else blw_clr)
            patch.set_label("Above strike" if bin_center > self.option.K else "Below strike")

        # Limits, labels etc.
        ax_hist.set_ylim(ax_paths.get_ylim())  # Match y axis limits.        
        ax_hist.set_xlabel("Price Distribution (Normalised)")

        # Create legend for the histogram.
        handles, labels = ax_hist.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_hist.legend(by_label.values(), by_label.keys(), loc="upper right")

    def animate_simulations(
        self,
        filename: str = "monte-carlo.gif",
        interval: int = 100,
        max_frames: int | None = None
    ) -> None:
        """Animate the paths being added and the histogram updating, then save as a GIF."""
        if self.results is None or self.avg_path is None or self.num_paths == 0:
            raise RuntimeError("Run run_simulations() before animating.")

        frames = 500#max_frames or min(self.num_paths, 400)
        
        fig, (ax_paths, ax_hist) = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.1},
        )

        def init():
            
            ax_paths.set_xlim(0, self.option.T)
            ax_paths.set_xlabel("Time, $t$ [Years]")
            ax_paths.set_ylabel("Stock Price, $S(t)$ [\\$]")
            ax_paths.set_title(
                f"Monte Carlo: Spot = \\${self.option.S0:.2f}, "
                + f"r = {self.option.r*100:.1f}%, "
                + f"σ = {self.option.sigma*100:.1f}%"
            )

            ax_hist.set_xlabel("Price Distribution (Normalised)")
            ax_hist.set_ylim(ax_paths.get_ylim())
            return []

        def update(frame):
            N = frame + 1
            ax_paths.cla()
            ax_hist.cla()

            above_strike = self.results[:N, -1] >= self.option.K
            colors = np.where(above_strike, "green", "red")

            for i in range(N):
                ax_paths.plot(
                    self.t,
                    self.results[i],
                    color=colors[i],
                    lw=0.7,
                    alpha=0.6,
                )

            ax_paths.plot([], [] , color="green", label=r"Paths ends $\bf{above}$ strike price")
            ax_paths.plot([], [] , color="red", label=r"Paths ends $\bf{below}$ strike price")
            ax_paths.axhline(self.option.K, color="k", linestyle="--", lw=2)
            ax_paths.set_xlim(0, self.option.T)
            ax_paths.set_ylim(self.results.min() * 0.9, self.results.max() * 1.1)
            ax_paths.set_xlabel("Time, $t$ [Years]")
            ax_paths.set_ylabel("Stock Price, $S(t)$ [\\$]")
            ax_paths.set_title(
                f"Monte Carlo: Spot = \\${self.option.S0:.2f}, "
                + f"r = {self.option.r*100:.1f}%, "
                + f"σ = {self.option.sigma*100:.1f}%"
            )
            evolving_avg = np.mean(self.results[:N, :], axis=0)
            ax_paths.plot(
                self.t,
                evolving_avg,
                color="blue",
                lw=2,
                label=f"Avg. path, $E[S(t=T)]$ = \\${evolving_avg[-1]:.2f}"
            )
            ax_paths.legend(loc="upper left")

            num_bins = 200 if 200 < round(self.num_paths / 10) else round(self.num_paths / 10)
            final_prices = self.results[:N, -1]
            _, bins_edge, patches = ax_hist.hist(
                final_prices,
                bins=num_bins,
                orientation="horizontal",
                color="lightgray",
                density=True
            )

            bin_centers = 0.5 * (bins_edge[:-1] + bins_edge[1:])
            for patch, center in zip(patches, bin_centers):
                patch.set_facecolor("green" if center >= self.option.K else "red")

            ax_hist.axhline(self.option.K, color="k", linestyle="--", lw=2)
            ax_hist.set_ylim(ax_paths.get_ylim())
            ax_hist.set_xlabel("Price Distribution (Normalised)")
            ax_hist.legend(
                [plt.Line2D([], [], color="green", lw=8),
                 plt.Line2D([], [], color="red", lw=8),
                 plt.Line2D([], [], color="k", linestyle="--", lw=2)],
                ["Above strike", "Below strike", f"Strike = \\${self.option.K:.2f}"],
                loc="upper right",
            )

            return ax_paths.lines + ax_hist.patches

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            interval=interval,
            blit=False,
            repeat=False,
        )

        writer = animation.PillowWriter(fps=1000 // interval)
        ani.save(filename, writer=writer)
        plt.close(fig)
