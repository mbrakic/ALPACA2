import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.optimize import minimize, NonlinearConstraint, brentq
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import warnings
from dataclasses import dataclass, field
import statsmodels.api as sm
from cyipopt import minimize_ipopt
from .ThresholdAnalyzer import ThresholdAnalyzer 

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Try to import sklearn components, but don't make it a hard dependency.
try:
    from sklearn.utils import resample
except ImportError:
    warnings.warn(
        "scikit-learn not found. Please install it (`pip install scikit-learn`) "
        "to use the richardson_extrapolation method."
    )
    resample = None


@dataclass
class GPDParameters:
    """Container for GPD distribution parameters, focused on endpoint estimation."""
    shape: float         # xi parameter
    scale: float         # sigma parameter
    threshold: float     # u, the threshold
    location: float = 0.0  # loc parameter, fixed to 0 for excesses

    @property
    def endpoint(self) -> float:
        """
        Calculate the upper endpoint of the original data's distribution.
        This is finite only if shape (xi) < 0.
        """
        if self.shape < 0:
            return self.threshold - (self.scale / self.shape)
        return np.inf

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple format for scipy functions."""
        return (self.shape, self.location, self.scale)

    def is_valid(self) -> bool:
        """Check if parameters are valid for a GPD."""
        return self.scale > 0 and self.shape < 0


@dataclass
class POTResults:
    """Container for initial POT analysis results."""
    L_low: Optional[float]
    L_high: Optional[float]
    N_total: int
    N_excess: int
    gamma: float
    threshold: float
    epsilon_dkw: float
    D_NS: float
    central_gpd_params: Optional[GPDParameters]
    gpd_fit_failed: bool
    error_message: Optional[str] = None


@dataclass
class BMAResults:
    """Container for Bayesian Model Averaging results."""
    bma_endpoint: float
    weighted_parameters: pd.DataFrame


@dataclass
class alpacaAnalysis:
    """Container for the comprehensive results of the alpaca full analysis."""
    pot_results: POTResults
    acceptable_params_df: pd.DataFrame
    bma_results: Optional[BMAResults]
    final_L_low: float
    final_L_high: float
    max_endpoint_from_optimization: Optional[float] = None
    max_endpoint_from_search: Optional[float] = None
    best_empirical_fit_params: Optional[GPDParameters] = None
    analysis_halted_reason: Optional[str] = None

    @property
    def optimization_succeeded(self) -> bool:
        """
        Returns True ONLY if the constrained optimization step succeeded 
        and found a valid, finite endpoint.
        Ignores results from the fallback parameter search.
        """
        if self.analysis_halted_reason is not None:
            return False
        
        # Must have a result from optimization (not None)
        if self.max_endpoint_from_optimization is None:
            return False
            
        # Must be finite
        if np.isinf(self.max_endpoint_from_optimization):
            return False
            
        return True


class ALPACA:
    """
    ALPACA: A Peaks-Over-Threshold (POT) analysis tool using the
    Generalized Pareto Distribution (GPD) to estimate a finite data endpoint.
    """

    # Class constants
    MIN_SCALE_THRESHOLD = 1e-9
    MIN_NEGATIVE_SHAPE = -1e-4 # Enforce negativity
    OPTIMIZATION_TOLERANCE = 1e-9

    @staticmethod
    def calculate_dkw_epsilon(N: int, gamma: float) -> float:
        """Calculates the DKW (Dvoretzky-Kiefer-Wolfowitz) epsilon value."""
        if N <= 0: return np.inf
        if not (0 < gamma < 1): raise ValueError("Gamma must be between 0 and 1")
        return np.sqrt(np.log(2.0 / gamma) / (2.0 * N))

    @classmethod
    def fit_gpd_parameters(
        cls, excesses: np.ndarray, threshold: float
    ) -> Tuple[Optional[GPDParameters], float]:
        """
        Fit GPD parameters to excess data and calculate goodness-of-fit.
        The location parameter is fixed to 0.
        """
        if len(excesses) < 3:
            return None, np.inf

        try:
            shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
            if scale <= cls.MIN_SCALE_THRESHOLD or shape >= 0:
                return None, np.inf
            gpd_params = GPDParameters(shape=shape, scale=scale, threshold=threshold)
            D_NS, _ = stats.kstest(excesses, 'genpareto', args=gpd_params.to_tuple())
            return gpd_params, D_NS
        except Exception:
            return None, np.inf
    
    @classmethod 
    def estimate_threshold(cls, data: np.ndarray, coords: np.ndarray, min_distance: int = 1,
                        significance_level: float = 0.05, starting_quantile: float = 0.8, 
                        backend: str = 'pytorch', plot: bool = False, 
                        true_threshold: Optional[None] = None
                        ) -> float: 

        analyzer = ThresholdAnalyzer(data, coords)

        found_threshold = analyzer.determine_threshold(
            min_distance, significance_level, starting_quantile, backend, plot, true_threshold
        )
        
        return found_threshold

    @classmethod
    def estimate_initial_interval(
        cls, data: np.ndarray, threshold: float, gamma: float, verbose: bool = False
    ) -> POTResults:
        """Performs an initial Peaks-Over-Threshold analysis for the endpoint."""
        n_total = len(data)
        excesses = data[data > threshold] - threshold
        n_excess = len(excesses)

        epsilon = cls.calculate_dkw_epsilon(n_excess, gamma)
        gpd_params, D_NS = cls.fit_gpd_parameters(excesses, threshold)

        if gpd_params is None:
            return POTResults(L_low=np.max(data), L_high=np.max(data),
                              N_total=n_total, N_excess=n_excess, gamma=gamma,
                              threshold=threshold, epsilon_dkw=epsilon, D_NS=D_NS,
                              central_gpd_params=None, gpd_fit_failed=True, error_message="GPD fitting failed")

        if D_NS > epsilon:
            reason = (f"Initial GPD fit (KS={D_NS:.4f}) is outside the "
                      f"DKW confidence band (eps={epsilon:.4f}).")
            return POTResults(L_low=np.max(data), L_high=np.max(data),
                              N_total=n_total, N_excess=n_excess, gamma=gamma,
                              threshold=threshold, epsilon_dkw=epsilon, D_NS=D_NS,
                              central_gpd_params=gpd_params, gpd_fit_failed=True, error_message=reason)

        L_high_initial = gpd_params.endpoint
        results = POTResults(
            L_low=np.max(data), L_high=L_high_initial,
            N_total=n_total, N_excess=n_excess, gamma=gamma, threshold=threshold,
            epsilon_dkw=epsilon, D_NS=D_NS, central_gpd_params=gpd_params, gpd_fit_failed=False
        )
        if verbose: cls._print_pot_results(results)
        return results

    @staticmethod
    def _print_pot_results(results: POTResults) -> None:
        """Print detailed initial POT analysis results."""
        print("\n" + "="*50)
        print("INITIAL POT ANALYSIS (ENDPOINT)")
        print("="*50)
        print(f"Total Samples (N_total): {results.N_total}")
        print(f"Threshold (u): {results.threshold:.4f}")
        print(f"Number of Excesses (N_u): {results.N_excess}")
        print(f"Confidence (1-gamma): {1-results.gamma:.2f}")
        print(f"DKW Epsilon (for excesses): {results.epsilon_dkw:.6f}")
        print(f"Kolmogorov-Smirnov D_NS (GPD fit): {results.D_NS:.6f}")
        if results.central_gpd_params:
            params = results.central_gpd_params
            print(f"Central GPD Fit: Shape={params.shape:.4f}, Scale={params.scale:.4f}")
            print(f"--> Initial Endpoint Estimate: {params.endpoint:.4f}")
        print("="*50 + "\n")

    @classmethod
    def _is_fully_within_bounds(cls, test_params: GPDParameters, ecdf_excesses: ECDF, epsilon: float) -> bool:
        """Check if a candidate GPD CDF is entirely within the confidence bands."""
        x_domain = ecdf_excesses.x
        lower_band = ecdf_excesses.y - epsilon
        upper_band = ecdf_excesses.y + epsilon
        test_cdf = stats.genpareto.cdf(x_domain, *test_params.to_tuple())
        return np.all((test_cdf >= lower_band - cls.OPTIMIZATION_TOLERANCE) & (test_cdf <= upper_band + cls.OPTIMIZATION_TOLERANCE))

    @classmethod
    def find_max_endpoint_by_optimization(cls, central_gpd_params: GPDParameters, ecdf: ECDF, epsilon: float, verbose: bool = True) -> Optional[GPDParameters]:
        """
        Finds the largest finite endpoint GPD that fits within the DKW confidence bands,
        starting the search from the provided central fit's endpoint.
        """
        if verbose:
            print("\n--- Starting Root-Finding to Maximize Endpoint ---")

        # 1. Prepare Data
        threshold = central_gpd_params.threshold
        max_excess = ecdf.x[-1]
        excesses = ecdf.x[1:] # first value is -inf, quirk of stats package. 
        y_values = ecdf.y[1:] # similarly here, first value is always zero

        # 2. Pre-calculate DKW Log-Survival Bounds
        # S(x) bounds are [1 - U, 1 - L]
        L = np.maximum(0.0, y_values - epsilon) # clipped lower band for all i
        U = np.minimum(1.0, y_values + epsilon) # clipped upper band for all i
        
        log_S_lower = np.log(1.0 - U + 1e-15)  # From U (strictest floor for alpha)
        log_S_upper = np.log(1.0 - L + 1e-15)  # From L (strictest ceiling for alpha)

        # CHECK 1: Is Infinite Endpoint (Gumbel/Exponential) Feasible?
        # ---------------------------------------------------------
        # For shape=0, CDF is 1 - exp(-x/sigma).
        # Bounds imply: -x / ln(1-U) <= sigma <= -x / ln(1-L)
        
        denom_min = -log_S_lower # -ln(1-U)
        denom_max = -log_S_upper # -ln(1-L)
        
        # Handle division by zero for denom_max if L=0
        denom_max = np.maximum(denom_max, 1e-15)

        local_min_sigmas = excesses / denom_min
        local_max_sigmas = excesses / denom_max
        
        # Intersection of all local constraints
        required_min_sigma = np.max(local_min_sigmas)
        allowed_max_sigma = np.min(local_max_sigmas)
        
        if required_min_sigma <= allowed_max_sigma:
            if verbose:
                print(f">> A priori check passed: Infinite endpoint feasible.")
                print(f"   Valid Scale range: [{required_min_sigma:.4f}, {allowed_max_sigma:.4f}]")
            
            # Pick scale closest to central fit, clamped to valid range
            best_sigma = np.clip(central_gpd_params.scale, required_min_sigma, allowed_max_sigma)
            
            return type(central_gpd_params)(shape=0.0, scale=best_sigma, threshold=threshold)

        # 3. Gap Function: (Max Allowed Alpha) - (Min Required Alpha)
        def constraint_gap(mu_candidate):
            if mu_candidate <= max_excess: 
                return -1.0
                
            val = 1.0 - ((excesses + 1e-9) / mu_candidate) 
            log_dist = np.log(np.maximum(val, 1e-15)) # Negative values
            
            # Note: Division by negative log_dist flips inequalities
            # Alpha_Max corresponds to log_S_lower (derived from U)
            # Alpha_Min corresponds to log_S_upper (derived from L)
            alpha_max_global = np.min(log_S_lower / log_dist)
            alpha_min_global = np.max(log_S_upper / log_dist)
            
            return alpha_max_global - alpha_min_global

        # 4. Bracket the Root
        # We assume central_gpd_params is valid (Gap > 0). 
        # We expand outwards to find where Gap < 0.
        
        # If central fit is infinite/exponential (shape >= 0), we are already at the limit.
        if central_gpd_params.shape >= 0:
            if verbose: print(">> Central fit is already infinite/exponential.")
            return central_gpd_params

        current_endpoint = -central_gpd_params.scale / central_gpd_params.shape
        
        lower_bracket = current_endpoint
        upper_bracket = current_endpoint * 2.0
        print('ub', upper_bracket)
        search_limit = max_excess * (2**12) 
        print('sl', search_limit)
        
        is_infinite_feasible = False
        
        # Expand until we break constraints or hit infinity
        while True:
            if upper_bracket > search_limit:
                print('ub > sl')
                is_infinite_feasible = True
                break
                
            if constraint_gap(upper_bracket) < 0:
                print('crossing')
                break # Found the crossing
            
            print('looping, gap is', constraint_gap(upper_bracket))
            lower_bracket = upper_bracket
            upper_bracket *= 2.0

        # 5. Solve
        if is_infinite_feasible:
            print(lower_bracket,upper_bracket, constraint_gap(upper_bracket))
            if verbose: print(">> Infinite endpoint is feasible.")
            return type(central_gpd_params)(shape=0.0, scale=central_gpd_params.scale, threshold=threshold)

        try:
            mu_opt = brentq(constraint_gap, lower_bracket, upper_bracket, xtol=1e-5)
        except ValueError:
            if verbose: warnings.warn("Root finding failed to converge.")
            return None

        # 6. Reconstruct Parameters
        log_dist_opt = np.log(1.0 - excesses / mu_opt)
        alpha_final = np.max(log_S_upper / log_dist_opt) # Pick the binding alpha
        
        xi_final = -1.0 / alpha_final
        sigma_final = -mu_opt * xi_final
        
        if verbose:
            print(f"Optimization successful. Max Endpoint: {threshold + mu_opt:.4f}")

        return type(central_gpd_params)(shape=xi_final, scale=sigma_final, threshold=threshold)

    @classmethod
    def find_acceptable_parameter_space(cls, central_gpd_params: GPDParameters, ecdf: ECDF,
                                        epsilon: float, n_samples: int = 10000,
                                        search_multipliers: Tuple[float, float] = (0.9, 1.1),
                                        use_fine_graining: bool = False,
                                        fine_grain_samples: int = 20000,
                                        fine_grain_multiplier: float = 0.05,
                                        verbose: bool = True) -> pd.DataFrame:
        """
        Find GPD parameters that satisfy ECDF-based confidence tube constraints
        via random search. Note here that ecdf refers to ecdf of the excesses.
        """
        def search_loop(num_to_sample, search_bounds, existing_params):
            newly_valid_params = []
            for _ in range(num_to_sample):
                scale = np.random.uniform(*search_bounds['scale'])
                shape = np.random.uniform(*search_bounds['shape'])
                if scale <= cls.MIN_SCALE_THRESHOLD or shape >= cls.MIN_NEGATIVE_SHAPE:
                    continue
                test_params = GPDParameters(shape=shape, scale=scale, threshold=central_gpd_params.threshold)
                if cls._is_fully_within_bounds(test_params, ecdf, epsilon):
                    newly_valid_params.append({'shape': shape, 'scale': scale, 'endpoint': test_params.endpoint})
            return existing_params + newly_valid_params

        if n_samples > 0 and verbose:
            print(f"Starting initial search for acceptable parameter space ({n_samples:,} samples)...")
        initial_bounds = {
            'scale': (central_gpd_params.scale * search_multipliers[0], central_gpd_params.scale * search_multipliers[1]),
            'shape': (-2.0, cls.MIN_NEGATIVE_SHAPE)
        }
        valid_params = search_loop(n_samples, initial_bounds, [])
        if n_samples > 0 and verbose:
            print(f"Found {len(valid_params):,} valid parameter sets in initial search.")

        if use_fine_graining and valid_params:
            if fine_grain_samples > 0 and verbose:
                print(f"\nStarting fine-graining search ({fine_grain_samples:,} samples)...")
            best_initial_df = pd.DataFrame(valid_params)
            best_fit = best_initial_df.loc[best_initial_df['endpoint'].idxmax()]
            if fine_grain_samples > 0 and verbose:
                print(f"Centering fine-grain search around best initial endpoint: {best_fit.endpoint:.4f}")
            fine_grain_bounds = {
                'scale': (best_fit['scale'] * (1 - fine_grain_multiplier), best_fit['scale'] * (1 + fine_grain_multiplier)),
                'shape': (best_fit['shape'] * (1 - fine_grain_multiplier), best_fit['shape'] * (1 + fine_grain_multiplier))
            }
            valid_params = search_loop(fine_grain_samples, fine_grain_bounds, valid_params)
            if fine_grain_samples > 0 and verbose:
                print(f"Total valid sets after fine-graining: {len(valid_params):,}")

        return pd.DataFrame(valid_params).drop_duplicates()

    @classmethod
    def calculate_bayesian_model_average(
        cls, excesses: np.ndarray, acceptable_params_df: pd.DataFrame, verbose: bool = True
    ) -> Optional[BMAResults]:
        """Calculates a Bayesian Model Average (BMA) for the GPD endpoint."""
        if acceptable_params_df.empty:
            return None
        params_with_weights = acceptable_params_df.copy()
        log_likelihoods = np.array([
            np.sum(stats.genpareto.logpdf(excesses, c=c, scale=s, loc=0))
            for c, s in zip(params_with_weights['shape'], params_with_weights['scale'])
        ])
        finite_mask = np.isfinite(log_likelihoods)
        params_with_weights = params_with_weights[finite_mask]
        log_likelihoods = log_likelihoods[finite_mask]
        if params_with_weights.empty: return None
        max_log_like = np.max(log_likelihoods)
        weights = np.exp(log_likelihoods - max_log_like)
        weights /= np.sum(weights)
        params_with_weights['weight'] = weights
        bma_endpoint = np.sum(params_with_weights['weight'] * params_with_weights['endpoint'])
        if verbose:
            print(f"\nBayesian Model Averaging successful. BMA Endpoint: {bma_endpoint:.4f}")
        return BMAResults(bma_endpoint=bma_endpoint, weighted_parameters=params_with_weights)

    @staticmethod
    def plot_estimation_results(
        excesses: np.ndarray, pot_results: POTResults,
        additional_fits: Optional[Dict] = None,
        bma_endpoint: Optional[float] = None,
        title_suffix: str = "", figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """Create a visualization of the POT analysis results for the excesses."""
        if pot_results.gpd_fit_failed: return
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=figsize)
        ecdf = ECDF(excesses)
        plt.step(ecdf.x, ecdf.y, where='post', label='Empirical CDF of Excesses', color='royalblue', alpha=0.8, lw=2)
        x_vals = np.linspace(0, np.max(excesses) * 1.05, 1000)
        epsilon = pot_results.epsilon_dkw
        plt.fill_between(ecdf.x, ecdf.y - epsilon, ecdf.y + epsilon, step='post', color='red', alpha=0.1, label=f'DKW Band')
        if pot_results.central_gpd_params:
            cdf_vals = stats.genpareto.cdf(x_vals, *pot_results.central_gpd_params.to_tuple())
            plt.plot(x_vals, cdf_vals, label='Central GPD Fit', color='green', linestyle='--', lw=2)
        if additional_fits and 'Max Endpoint Empirical Fit' in additional_fits:
            params, _, _, _ = additional_fits['Max Endpoint Empirical Fit']
            cdf_vals = stats.genpareto.cdf(x_vals, *params)
            plt.plot(x_vals, cdf_vals, label='Max Endpoint Fit', linestyle='-', alpha=0.9, lw=2.5, color='purple')
        info_text = (f"Threshold: {pot_results.threshold:.4f}\n"
                     f"Excesses: {pot_results.N_excess}/{pot_results.N_total}\n\n"
                     f"--- Endpoint Estimates ---\n"
                     f"Initial GPD Fit: {pot_results.L_high:.4f}\n")
        if bma_endpoint: info_text += f"BMA Endpoint: {bma_endpoint:.4f}\n"
        if additional_fits and 'Max Endpoint Empirical Fit' in additional_fits:
            _, endpoint, _, _ = additional_fits['Max Endpoint Empirical Fit']
            info_text += f"Max Endpoint from Search: {endpoint:.4f}\n"
        plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6))
        plt.title(f'POT Analysis for Endpoint Estimation {title_suffix}'.strip(), fontsize=16, pad=20)
        plt.xlabel(f'Excess over threshold u={pot_results.threshold:.2f}', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, which='both', linestyle=':', linewidth=0.5)
        plt.ylim(-0.05, 1.05)
        plt.xlim(0, x_vals[-1])
        plt.tight_layout()
        print("\nPlot generated successfully.")

    @classmethod
    def run_full_analysis(cls, data: np.ndarray, coords: np.ndarray, 
                          gamma: float, determine_optimal_threshold = True,
                          use_parameter_search: bool = True, use_fine_graining: bool = True,
                          n_search_samples: int = 20000, show_plot: bool = False,
                          verbose: bool = True) -> alpacaAnalysis:
        """Runs the complete analysis pipeline and returns a summary report object."""
        # --- Initial Data Checks ---
        if len(data) == 0:
            return alpacaAnalysis(
                pot_results=POTResults(L_low=None, L_high=None, N_total=0, N_excess=0, gamma=gamma, threshold=0, epsilon_dkw=np.inf, D_NS=np.inf, central_gpd_params=None, gpd_fit_failed=True, error_message="Empty data array"),
                acceptable_params_df=pd.DataFrame(), bma_results=None,
                final_L_low=-np.inf, final_L_high=np.inf, analysis_halted_reason="Empty data array"
            )

        # --- Threshold Estimation ---
        if determine_optimal_threshold:
            threshold = cls.estimate_threshold(data, coords, plot=show_plot)
        else:
            threshold = np.quantile(data, 0.96)

        # analyzer = ThresholdAnalyzer(data, coords) 
        # # data = analyzer._extract_independent_peaks(1)
        # data = analyzer._extract_spatial_peaks_scipy(coords, data, radius=0.01)

        pot_results = cls.estimate_initial_interval(data, threshold, gamma, verbose=verbose)

        # --- Early Exit Checks ---
        if pot_results.gpd_fit_failed or not pot_results.central_gpd_params or pot_results.central_gpd_params.shape >= 0:
            reason = pot_results.error_message or "Initial GPD fit implies an infinite endpoint (shape >= 0)."
            if verbose: print(f"\nAnalysis HALTED: {reason}")
            return alpacaAnalysis(
                pot_results=pot_results, acceptable_params_df=pd.DataFrame(), bma_results=None,
                final_L_low=np.max(data) if data.size > 0 else -np.inf,
                final_L_high=pot_results.L_high if pot_results.L_high is not None else np.inf,
                analysis_halted_reason=reason
            )

        # --- Main Analysis ---
        excesses = data[data > threshold] - threshold
        ecdf = ECDF(excesses)

        optimized_params = cls.find_max_endpoint_by_optimization(
            pot_results.central_gpd_params, ecdf, pot_results.epsilon_dkw, verbose=verbose
        )

        print('optimised params')
        print(optimized_params)

        # --- Initialize result containers ---
        acceptable_params_df = pd.DataFrame()
        bma_results = None
        max_endpoint_from_search = None
        best_empirical_fit_params = pot_results.central_gpd_params
        final_L_high = pot_results.L_high
        max_opt_endpoint = None

        if optimized_params:
            max_opt_endpoint = optimized_params.endpoint
            best_empirical_fit_params = optimized_params
            final_L_high = max_opt_endpoint
        elif use_parameter_search: # Only run search if optimization fails
            acceptable_params_df = cls.find_acceptable_parameter_space(
                pot_results.central_gpd_params, ecdf, pot_results.epsilon_dkw,
                n_samples=n_search_samples, use_fine_graining=use_fine_graining, verbose=verbose
            )
            if not acceptable_params_df.empty:
                bma_results = cls.calculate_bayesian_model_average(excesses, acceptable_params_df, verbose=verbose)
                best_fit_row = acceptable_params_df.loc[acceptable_params_df['endpoint'].idxmax()]
                max_endpoint_from_search = best_fit_row['endpoint']
                best_empirical_fit_params = GPDParameters(shape=best_fit_row['shape'], scale=best_fit_row['scale'], threshold=threshold)
                # Update final_L_high with the best of all estimates found
                final_L_high = max(final_L_high, max_endpoint_from_search)
                if bma_results:
                    final_L_high = max(final_L_high, bma_results.bma_endpoint)

        # <<< CHANGE 2: ADDED COMBINED FAILURE CHECK >>>
        if optimized_params is None and acceptable_params_df.empty:
            reason = "Optimization failed and parameter search found no valid models within the DKW band."
            if verbose: print(f"\nAnalysis HALTED: {reason}")
            return alpacaAnalysis(
                pot_results=pot_results,
                acceptable_params_df=pd.DataFrame(),
                bma_results=None,
                final_L_low=pot_results.L_low,
                final_L_high=pot_results.L_high, # Revert to initial estimate
                analysis_halted_reason=reason
            )

        # --- Final Reporting and Plotting ---
        if verbose:
            print("\n" + "="*60)
            print("FINAL POT ANALYSIS SUMMARY (ENDPOINT)".center(60))
            print("="*60)
            if max_opt_endpoint: print(f"Max Endpoint from Optimization: {max_opt_endpoint:.4f}")
            if use_parameter_search:
                if max_endpoint_from_search is not None: print(f"Max Endpoint from Search: {max_endpoint_from_search:.4f}")
                if bma_results: print(f"BMA Endpoint: {bma_results.bma_endpoint:.4f}")
            print("\n" + "-"*60)
            print(f"Final Reported PAC Interval: [{pot_results.L_low:.4f}, {final_L_high:.4f}]")
            print("="*60)

        if show_plot:
            additional_fits = {}
            if best_empirical_fit_params:
                additional_fits['Max Endpoint Empirical Fit'] = (
                    best_empirical_fit_params.to_tuple(), best_empirical_fit_params.endpoint, '-', 'purple'
                )

            bma_val = bma_results.bma_endpoint if bma_results else None
            cls.plot_estimation_results(excesses=excesses, pot_results=pot_results,
                                        additional_fits=additional_fits,
                                        bma_endpoint=bma_val)
            plt.show()

        return alpacaAnalysis(
            pot_results=pot_results,
            acceptable_params_df=acceptable_params_df,
            bma_results=bma_results,
            final_L_low=pot_results.L_low,
            final_L_high=final_L_high,
            max_endpoint_from_optimization=max_opt_endpoint,
            max_endpoint_from_search=max_endpoint_from_search,
            best_empirical_fit_params=best_empirical_fit_params,
        )



    @classmethod
    def _analyze_excesses_for_extrapolation(cls, excesses: np.ndarray, threshold: float, gamma: float, n_search_samples: int) -> Dict:
        """
        A streamlined analysis function specifically for the extrapolation loop.
        Returns L_low (sample max), L_high (theoretical max), and BMA endpoint.
        """
        n_excess = len(excesses)
        if n_excess < 3:
            return {'L_low': np.nan, 'L_high': np.nan, 'BMA': np.nan}

        l_low = threshold + np.max(excesses)
        epsilon = cls.calculate_dkw_epsilon(n_excess, gamma)
        central_gpd_params, _ = cls.fit_gpd_parameters(excesses, threshold)

        if central_gpd_params is None:
            return {'L_low': l_low, 'L_high': np.nan, 'BMA': np.nan}

        final_L_high = central_gpd_params.endpoint
        bma_endpoint = np.nan

        ecdf = ECDF(excesses)

        optimized_params = cls.find_max_endpoint_by_optimization(
            central_gpd_params, ecdf, epsilon, verbose=False
        )
        if optimized_params:
            max_opt_endpoint = optimized_params.endpoint 
            final_L_high = max(final_L_high, max_opt_endpoint) 
            bma_endpoint = None
        else:
            acceptable_params = cls.find_acceptable_parameter_space(
                central_gpd_params, ecdf, epsilon, n_samples=n_search_samples,
                use_fine_graining=True, fine_grain_samples=n_search_samples, verbose=False
            )

            if not acceptable_params.empty:
                max_endpoint_from_search = acceptable_params['endpoint'].max()
                final_L_high = max(final_L_high, max_endpoint_from_search)
                
                bma_results = cls.calculate_bayesian_model_average(excesses, acceptable_params, verbose=False)
                if bma_results:
                    bma_endpoint = bma_results.bma_endpoint
                    final_L_high = max(final_L_high, bma_endpoint)

        return {'L_low': l_low, 'L_high': final_L_high, 'BMA': bma_endpoint}


    @classmethod
    def richardson_extrapolation(
        cls,
        master_data: np.ndarray,
        gamma: float,
        num_partitions: int = 8,
        num_repeats: int = 1,
        n_search_samples: int = 10000,
        bootstrap_samples: int = 1000,
        min_excess_count: int = 400,
        outlier_rejection_iqr_scale: Optional[float] = 1.5,
        true_endpoint: Optional[float] = None,
        show_plot: bool = True,
    ) -> Dict:
        """
        Performs Richardson extrapolation on endpoint estimates.

        This method partitions the data, calculates endpoint estimates for each
        partition, and then extrapolates to an infinite sample size. It can
        optionally perform a parameter search and Bayesian Model Averaging (BMA)
        for more robust estimates.

        Args:
            master_data: A 2D numpy array where each column is a repetition of an experiment.
            threshold_quantile: The quantile used to set the threshold for POT.
            gamma: The confidence level parameter (e.g., 0.05 for 95% confidence).
            num_partitions: The number of data partitions to create for extrapolation.
            num_repeats: The number of columns in `master_data` to process.
            n_search_samples: The number of samples for the parameter search. If 0,
                              the search and BMA are skipped.
            bootstrap_samples: The number of bootstrap samples for error estimation.
            min_excess_count: The minimum number of excesses required for a partition.
            outlier_rejection_iqr_scale: The IQR scale for robust WLS outlier rejection.
            true_endpoint: Optional. If provided, it's plotted for comparison.
            show_plot: Whether to display the final extrapolation plot.

        Returns:
            A dictionary containing the extrapolation results for each estimate type.
        """
        print(f"\n--- Starting Richardson Extrapolation ({num_partitions} partitions, {num_repeats} repeats) ---")

        if master_data.ndim == 1:
            master_data = master_data.reshape(-1, 1)
            if num_repeats > 1:
                warnings.warn("master_data is 1D but num_repeats > 1. Extrapolation will only run once.")
                num_repeats = 1

        if master_data.shape[1] < num_repeats:
            raise ValueError(f"master_data has {master_data.shape[1]} columns, but {num_repeats} repeats were requested.")

        L_low_repeats, L_high_repeats, BMA_repeats = [], [], []

        first_rep_data = master_data[:, 0]
        threshold = cls.estimate_threshold(first_rep_data)
        all_excesses_first_rep = first_rep_data[first_rep_data > threshold] - threshold
        max_n_excess = len(all_excesses_first_rep)
        if max_n_excess < min_excess_count * 2:
            raise ValueError(f"Not enough excesses ({max_n_excess}) in the first repetition to define partitions.")
        partition_sizes = np.linspace(min_excess_count, max_n_excess, num_partitions, dtype=int)

        for i in range(num_repeats):
            print(f"\n--- Starting Repetition {i+1}/{num_repeats} ---")
            current_data = master_data[:, i]
            threshold = cls.estimate_threshold(current_data)
            all_excesses = current_data[current_data > threshold] - threshold

            L_low_N, L_high_N, BMA_N = [], [], []
            for size in partition_sizes:
                if size > len(all_excesses):
                    L_low_N.append(np.nan)
                    L_high_N.append(np.nan)
                    BMA_N.append(np.nan)
                    continue

                partition_excesses = all_excesses[:size]
                analysis_results = cls._analyze_excesses_for_extrapolation(partition_excesses, threshold, gamma, n_search_samples)
                L_low_N.append(analysis_results['L_low'])
                L_high_N.append(analysis_results['L_high'])
                BMA_N.append(analysis_results['BMA'])

            L_low_repeats.append(L_low_N)
            L_high_repeats.append(L_high_N)
            BMA_repeats.append(BMA_N)

        if not L_high_repeats:
            raise RuntimeError("Extrapolation failed: No successful repetitions.")

        print("\n" + "="*80)
        print("Averaging results across all repetitions and fitting models...")
        L_low_avg = np.nanmean(np.array(L_low_repeats), axis=0)
        L_high_avg = np.nanmean(np.array(L_high_repeats), axis=0)
        
        df_data = {
            'N_excess': partition_sizes,
            '1/N_excess': 1.0 / partition_sizes,
            'L_low': L_low_avg,
            'L_high': L_high_avg,
        }
        if n_search_samples > 0:
            df_data['BMA'] = np.nanmean(np.array(BMA_repeats), axis=0)
        
        df = pd.DataFrame(df_data)

        print("\n--- Averaged Extrapolation Results per Partition ---")
        print(df.to_string())

        def _perform_robust_wls(data_df: pd.DataFrame, y_col: str, iqr_scale: Optional[float]):
            """Performs WLS with optional IQR outlier rejection."""
            clean_df = data_df.dropna(subset=[y_col, '1/N_excess', 'N_excess'])
            if len(clean_df) < 2: return None, pd.DataFrame(), pd.DataFrame()

            y = clean_df[y_col]
            X = sm.add_constant(clean_df['1/N_excess'])
            weights = clean_df['N_excess']
            model_initial = sm.WLS(y, X, weights=weights).fit()

            inlier_mask = pd.Series(True, index=y.index)
            if iqr_scale and len(y) > 3:
                residuals = model_initial.resid
                q1, q3 = residuals.quantile(0.25), residuals.quantile(0.75)
                iqr = q3 - q1
                inlier_mask = (residuals >= (q1 - iqr_scale * iqr)) & (residuals <= (q3 + iqr_scale * iqr))

            df_inliers = clean_df.loc[inlier_mask]
            df_outliers = clean_df.loc[~inlier_mask]

            if len(df_inliers) < 2: return model_initial, df_inliers, df_outliers

            final_model = sm.WLS(df_inliers[y_col], sm.add_constant(df_inliers['1/N_excess']), weights=df_inliers['N_excess']).fit()
            return final_model, df_inliers, df_outliers

        results = {}
        cols_to_extrapolate = ['L_low', 'L_high']
        if n_search_samples > 0:
            cols_to_extrapolate.append('BMA')

        for col in cols_to_extrapolate:
            if col not in df.columns or df[col].isnull().all(): continue
            print(f"\n--- Extrapolating for Averaged {col} ---")

            model, df_inliers, df_outliers = _perform_robust_wls(df, col, outlier_rejection_iqr_scale)

            if model is None or not all(k in model.params for k in ['const', '1/N_excess']):
                print("  -> Regression failed.")
                continue

            asymptotic_val, slope = model.params['const'], model.params['1/N_excess']
            results[col] = {'asymptotic_value': asymptotic_val, 'slope': slope, 'inliers': df_inliers, 'outliers': df_outliers}

            print(f"Identified {len(df_inliers)} inliers and {len(df_outliers)} outliers for {col}.")
            if not df_outliers.empty:
                outlier_N_values = df_outliers['N_excess'].to_numpy()
                print(f"  -> Removed points corresponding to N_excess: {outlier_N_values}")

            bootstrap_intercepts = []
            if bootstrap_samples > 0 and resample is not None and not df_inliers.empty:
                print(f"Bootstrapping from {len(df_inliers)} inlier points...")
                for _ in range(bootstrap_samples):
                    boot_df = resample(df_inliers)
                    if len(boot_df) < 2: continue
                    boot_y = boot_df[col]
                    boot_X = sm.add_constant(boot_df['1/N_excess'])
                    boot_weights = boot_df['N_excess']
                    try:
                        boot_model = sm.WLS(boot_y, boot_X, weights=boot_weights).fit()
                        if 'const' in boot_model.params:
                            bootstrap_intercepts.append(boot_model.params['const'])
                    except (np.linalg.LinAlgError, ValueError):
                        continue

                if bootstrap_intercepts:
                    error = np.std(bootstrap_intercepts)
                    results[col]['error'] = error
                    print(f"Asymptotic {col}: {asymptotic_val:.4f} \u00B1 {2*error:.4f} (2 std err)")
                else:
                    results[col]['error'] = np.nan
            else:
                results[col]['error'] = np.nan

        if show_plot:
            num_plots = len(results)
            if num_plots == 0:
                print("No results to plot.")
                return results

            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), sharex=True, constrained_layout=True)
            if num_plots == 1: axes = [axes]
            fig.suptitle('Robust Richardson Extrapolation (Averaged over Repetitions)', fontsize=20, y=1.03)

            for i, (col, res) in enumerate(results.items()):
                ax = axes[i]
                color_map = {'L_low': 'royalblue', 'L_high': 'red', 'BMA': 'green'}
                color = color_map.get(col, 'black')

                ax.scatter(res['inliers']['1/N_excess'], res['inliers'][col], label=f'Inliers for {col}', color=color, alpha=0.9, s=80, zorder=5)
                if not res['outliers'].empty:
                    ax.scatter(res['outliers']['1/N_excess'], res['outliers'][col], edgecolors=color, facecolors='none', s=100, label=f'Outliers for {col}', marker='o', linewidth=1.5, zorder=5)

                if not res['inliers'].empty:
                    x_fit = np.array([0, res['inliers']['1/N_excess'].max()])
                    ax.plot(x_fit, res['asymptotic_value'] + res['slope'] * x_fit, color=color, linestyle='--', label=f'WLS Fit on Inliers')

                ax.errorbar(0, res['asymptotic_value'], yerr=res.get('error', 0), fmt='X', color='black', capsize=7, markersize=14, label=f'Extrap. = {res["asymptotic_value"]:.4f} \u00B1 {res.get("error", 0):.4f}', zorder=10)

                if true_endpoint is not None:
                    ax.axhline(true_endpoint, color='grey', linestyle=':', label=f'True Endpoint = {true_endpoint:.4f}')

                ax.set_ylabel(f'Averaged {col} Estimate', fontsize=12)
                ax.legend(loc='best')
                ax.grid(True, which='both', linestyle=':')

            axes[-1].set_xlabel('$N_{excess}^{-1}$', fontsize=14)
            axes[-1].invert_xaxis()
            plt.show()

        return results

    @staticmethod
    def pretty_print(results: alpacaAnalysis) -> None:
        """Prints a formatted, human-readable summary of the LipPOT analysis results."""
        print("\n" + "="*80)
        print("LipPOT: Full Analysis Report".center(80))
        print("="*80)
        if results.analysis_halted_reason:
            print(f"\n[ANALYSIS HALTED]\nReason: {results.analysis_halted_reason}")
            print("="*80)
            return

        pac = results.pot_results
        print("\n--- [1] Analysis Configuration ---")
        print(f"Total Input Sample Size (N)                  : {pac.N_total}")
        print(f"Excess Sample Size (N_u)                     : {pac.N_excess} (at threshold u={pac.threshold:.4f})")
        print(f"Confidence Level (1-gamma)                   : {1 - pac.gamma:.3f} (gamma = {pac.gamma})")
        print(f"DKW Epsilon (Non-parametric band width)      : {pac.epsilon_dkw:.5f}")

        print("\n--- [2] Initial Central GPD Fit ---")
        if pac.central_gpd_params:
            params = pac.central_gpd_params
            print(f"Fit Quality (KS-Distance, D_NS)              : {pac.D_NS:.5f}")
            print(f"Parameters (shape, scale)                    : ({params.shape:.4f}, {params.scale:.4f})")
            if pac.D_NS > pac.epsilon_dkw:
                 print(f"NOTE: Initial GPD fit (D_NS={pac.D_NS:.4f}) is outside the DKW band (eps={pac.epsilon_dkw:.4f}).")
        else:
            print("Initial GPD fitting failed.")

        print("\n--- [3] Endpoint Estimation Details ---")
        if pac.central_gpd_params:
            print(f"Initial Endpoint (from central GPD fit)      : {pac.central_gpd_params.endpoint:.5f}")
        if results.max_endpoint_from_optimization is not None:
            print(f"Max Endpoint (from optimization)             : {results.max_endpoint_from_optimization:.5f}")
        else:
            print("Optimization to find max endpoint failed or was not performed.")

        if not results.acceptable_params_df.empty:
            print(f"\nParameter search found {len(results.acceptable_params_df):,} alternative valid models.")
            if results.max_endpoint_from_search is not None:
                print(f"Max Endpoint (from search)                   : {results.max_endpoint_from_search:.5f}")
            if results.bma_results and results.bma_results.bma_endpoint is not None:
                print(f"Bayesian Model Averaged (BMA) Endpoint       : {results.bma_results.bma_endpoint:.5f}")
        else:
            print("\nParameter search was not performed or found no valid models.")
        
        print("\n--- [4] Final PAC Confidence Interval ---")
        print(f"Interval: [{results.final_L_low:.5f}, {results.final_L_high:.5f}]")
        print(f"  - L_low is the maximum observed value from the samples.")
        print(f"  - L_high is the maximum of all credible endpoint estimates found.")
        print("\n" + "="*80)

def run_gpd_endpoint_example():
    """Run a comprehensive example for endpoint estimation using LipPOT."""
    print("Running LipPOT Endpoint Estimation Example")
    print("*"*80)
    true_gpd_params = (-0.25, 0, 2)
    sample_data = stats.genpareto.rvs(*true_gpd_params, size=50000)
    true_endpoint = true_gpd_params[1] - (true_gpd_params[2] / true_gpd_params[0])
    print(f"Generated {len(sample_data)} samples from a GPD({true_gpd_params[0]}, {true_gpd_params[2]}).")
    print(f"The true endpoint is: {true_endpoint:.4f}\n")
    analysis_results = LipPOT.run_full_analysis(
        data=sample_data, gamma=0.05,
        use_parameter_search=True, n_search_samples=10000, 
        show_plot=True, use_fine_graining=True, verbose=True
    )
    LipPOT.pretty_print(analysis_results)

def run_extrapolation_example():
    """Run an example of Richardson Extrapolation."""
    print("\n\n" + "*"*80)
    print("Running LipPOT Richardson Extrapolation Example")
    print("*"*80)
    true_gpd_params = (-0.25, 0, 2)
    true_endpoint = 0 - (2 / -0.25)
    print(f"Generating master dataset for extrapolation... True endpoint is {true_endpoint:.4f}")

    num_repeats = 0
    max_samples = 200000
    master_data = np.array([
        stats.genpareto.rvs(*true_gpd_params, size=max_samples) for _ in range(num_repeats)
    ]).T

    print("\n--- Running Extrapolation again BMA search ---")
    extrapolation_results_no_bma = LipPOT.richardson_extrapolation(
        master_data=master_data,
        gamma=0.05,
        num_partitions=8,
        num_repeats=num_repeats,
        n_search_samples=0, # Set to 0 to skip BMA extrapolation
        bootstrap_samples=500,
        min_excess_count=1000,
        outlier_rejection_iqr_scale=1.5,
        true_endpoint=true_endpoint,
        show_plot=True
    )
    # print("\n--- Final Extrapolation Results (No BMA) ---")
    # print(extrapolation_results_no_bma)


if __name__ == '__main__':
    run_gpd_endpoint_example()
    try:
        run_extrapolation_example()
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"\nCould not run extrapolation example: {e}")