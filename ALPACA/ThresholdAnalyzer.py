import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, chi2, norm
from scipy.signal import find_peaks
from scipy.spatial import KDTree
import time
from typing import Optional, Tuple
from tqdm import tqdm 

# --- Optional PyTorch Import ---
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

class ThresholdAnalyzer:
    """
    A class to determine the optimal threshold for extreme value analysis using
    Mean Residual Life plots and statistical tests.

    It supports both NumPy and PyTorch backends for computation.
    """

    def __init__(self, data: np.ndarray, coords: np.ndarray):
        """
        Initializes the ThresholdAnalyzer with the time series data.

        Args:
            data (np.ndarray): A 1D NumPy array of time series data.
        """
        # if not isinstance(data, np.ndarray) or data.ndim != 1:
        #     raise ValueError("Input data must be a 1D NumPy array.")
        self.data = data
        self.coords = coords
        self.pytorch_available = PYTORCH_AVAILABLE
        if self.pytorch_available:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[ThresholdAnalyzer] PyTorch is available. Using device: {self.device}")
        else:
            print("[ThresholdAnalyzer] PyTorch not found. Only NumPy backend is available.")

    def _extract_independent_peaks(self, min_distance_days: int) -> np.ndarray:

        """
        Extracts independent peaks from the time series data.

        Args:
            min_distance_days (int): The minimum number of days between peaks.

        Returns:
            np.ndarray: A sorted array of unique peak values.
        """
        peaks_indices, _ = find_peaks(self.data, distance=min_distance_days)
        peak_values = self.data[peaks_indices]
        unique_peaks = np.unique(peak_values)
        print(f"Found {len(unique_peaks)} independent peaks.")
        return np.sort(unique_peaks)

    def _extract_spatial_peaks_scipy(
            self, 
            coords: np.ndarray, 
            values: np.ndarray, 
            radius: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Extracts independent spatial peaks using Non-Maximum Suppression (NMS).
            This is the RECOMMENDED, fast O(N log N) method using SciPy's KDTree.
            
            Args:
                coords (np.ndarray): Shape (N, d) array of coordinates.
                values (np.ndarray): Shape (N,) array of corresponding values.
                radius (float): The suppression radius. Any point within this
                                distance of a peak will be suppressed.
                                
            Returns:
                Tuple[np.ndarray, np.ndarray]: 
                    - declustered_coords: Shape (M, d)
                    - declustered_values: Shape (M,)
            """
            print(f"[SciPy Backend] Starting spatial NMS... N={len(values)}, R={radius}")
            if coords.ndim != 2:
                raise ValueError(f"Coords must be 2D (N, d), but got {coords.shape}")
            if values.ndim != 1 or len(coords) != len(values):
                raise ValueError("Coords and Values must have matching N.")

            n_points = len(values)
            
            # 1. Build the KD-Tree on all coordinates for fast neighbor search
            start_tree = time.time()
            kdtree = KDTree(coords)
            print(f"  Built KD-Tree in {time.time() - start_tree:.2f}s")

            # 2. Sort points by value, descending
            sorted_indices = np.argsort(values)[::-1]
            
            # 3. Initialize suppression mask
            suppressed = np.zeros(n_points, dtype=bool)
            kept_indices = []

            # 4. Iterate from highest value to lowest
            start_loop = time.time()
            for i in tqdm(sorted_indices, desc="Spatial NMS (SciPy)"):
                if suppressed[i]:
                    continue
                    
                # This is an independent peak
                kept_indices.append(i)
                
                # Find all neighbors within the radius
                # This is the fast O(log N) query
                neighbor_indices = kdtree.query_ball_point(coords[i], radius)
                
                # Suppress them (neighbor_indices is a list of indices)
                suppressed[neighbor_indices] = True
                
            print(f"  NMS loop finished in {time.time() - start_loop:.2f}s")
            
            # 5. Return the declustered peaks
            kept_indices_arr = np.array(kept_indices)
            print(f"Declustering complete. Kept {len(kept_indices_arr)} / {n_points} peaks.")
            
            # Return in original (unsorted) order
            return np.sort(values[kept_indices_arr])

    def _test_threshold_slice_numpy(
        self,
        u_slice: np.ndarray,
        mean_excesses: np.ndarray,
        variances: np.ndarray,
        n_excesses_list: np.ndarray,
        significance_level: float
    ) -> bool:
        """
        Performs statistical tests on a slice of thresholds using NumPy.

        Args:
            u_slice (np.ndarray): Array of candidate thresholds.
            mean_excesses (np.ndarray): Mean excesses corresponding to the thresholds.
            variances (np.ndarray): Variances of excesses.
            n_excesses_list (np.ndarray): Number of excesses for each threshold.
            significance_level (float): The significance level for the statistical tests.

        Returns:
            bool: True if the slice is statistically valid, False otherwise.
        """
        try:
            weights = n_excesses_list / variances
            X = np.vstack([np.ones_like(u_slice), u_slice]).T
            Y = mean_excesses
            W = np.diag(weights)

            # Weighted Least Squares Calculation
            XTWX_inv = np.linalg.inv(X.T @ W @ X)
            beta = XTWX_inv @ (X.T @ W @ Y)
            residuals = Y - X @ beta

            # Goodness-of-fit and Outlier Tests
            P = X @ XTWX_inv @ X.T @ W
            wssr = np.sum(weights * residuals**2)

            n_points, n_params = len(Y), X.shape[1]
            sigma_sq_hat = wssr / (n_points - n_params)
            leverage = P[0, 0]

            if 1 - leverage <= 1e-9:
                return False

            studentized_residual_first = residuals[0] / np.sqrt(sigma_sq_hat * (1 - leverage) / weights[0])
            degrees_of_freedom = n_points - n_params

            # Critical values for the tests
            chi2_critical_value = chi2.ppf(1 - significance_level, df=degrees_of_freedom)
            z_critical_value = norm.ppf(1 - significance_level / 2)

            # A valid slice has a good fit (WSSR) and no significant outlier at the start
            return wssr < chi2_critical_value and np.abs(studentized_residual_first) < z_critical_value

        except np.linalg.LinAlgError:
            # Matrix may be singular if data is perfectly linear, etc.
            return False

    def _get_optimal_threshold_numpy(
        self,
        independent_peaks: np.ndarray,
        significance_level: float,
        starting_quantile: float
    ) -> float:
        """Identifies the optimal threshold using a binary search strategy with NumPy."""
        print("\n[NumPy Backend] Starting pre-computation for MRL statistics...")
        all_thresholds = independent_peaks
        mrl_means, mrl_vars, mrl_counts, valid_thresholds = [], [], [], []

        for u in all_thresholds:
            excesses = self.data[self.data > u]
            n_excesses = len(excesses)
            if n_excesses < 10:  # Need sufficient points for stable stats
                break
            valid_thresholds.append(u)
            mrl_means.append(np.mean(excesses - u))
            mrl_vars.append(np.var(excesses, ddof=1))
            mrl_counts.append(n_excesses)

        if not valid_thresholds:
            print("[NumPy Backend] No thresholds with sufficient data found.")
            return np.nan

        valid_thresholds = np.array(valid_thresholds)
        mrl_means = np.array(mrl_means)
        mrl_vars = np.array(mrl_vars)
        mrl_counts = np.array(mrl_counts)
        print(f"Pre-computation complete. Found {len(valid_thresholds)} candidate thresholds.")

        start_idx = np.searchsorted(valid_thresholds, np.quantile(independent_peaks, starting_quantile), side='left')
        print(f"Starting binary search from the {starting_quantile*100:.0f}th percentile of peaks.")
        print(f"Searching within {len(valid_thresholds) - start_idx} thresholds...")

        low, high = start_idx, len(valid_thresholds) - 1
        optimal_threshold_idx = -1

        while low <= high:
            mid = low + (high - low) // 2
            if len(valid_thresholds[mid:]) < 10:  # Ensure enough points for regression
                high = mid - 1
                continue

            is_valid = self._test_threshold_slice_numpy(
                u_slice=valid_thresholds[mid:],
                mean_excesses=mrl_means[mid:],
                variances=mrl_vars[mid:],
                n_excesses_list=mrl_counts[mid:],
                significance_level=significance_level
            )

            if is_valid:
                optimal_threshold_idx = mid  # This is a potential candidate
                high = mid - 1  # Try to find an even lower (better) threshold
            else:
                low = mid + 1  # This threshold is too low, search higher

        if optimal_threshold_idx != -1:
            optimal_threshold = valid_thresholds[optimal_threshold_idx]
            print(f"\n[NumPy Backend] Found optimal threshold: {optimal_threshold:.2f}.")
            return float(optimal_threshold)
        else:
            print("\n[NumPy Backend] No optimal threshold found that satisfies all criteria.")
            return np.nan

    def _test_threshold_slice_pytorch(
        self,
        u_slice: "torch.Tensor",
        mean_excesses_slice: "torch.Tensor",
        variances_slice: "torch.Tensor",
        n_excesses_slice: "torch.Tensor",
        significance_level: float
    ) -> bool:
        """
        Performs statistical tests on a slice of thresholds using PyTorch.

        Args:
            u_slice (torch.Tensor): Tensor of candidate thresholds.
            mean_excesses_slice (torch.Tensor): Mean excesses for the thresholds.
            variances_slice (torch.Tensor): Variances of excesses.
            n_excesses_slice (torch.Tensor): Number of excesses for each threshold.
            significance_level (float): The significance level for the tests.

        Returns:
            bool: True if the slice is statistically valid, False otherwise.
        """
        try:
            weights_t = n_excesses_slice / variances_slice
            X_t = torch.stack([torch.ones_like(u_slice), u_slice], dim=1)
            Y_t = mean_excesses_slice
            W_t = torch.diag(weights_t)

            # Weighted Least Squares using PyTorch
            XTW = X_t.T @ W_t
            beta_t = torch.linalg.solve(XTW @ X_t, XTW @ Y_t)
            residuals_t = Y_t - X_t @ beta_t

            # Goodness-of-fit and Outlier Tests
            P_t = X_t @ torch.linalg.solve(XTW @ X_t, X_t.T) @ W_t
            wssr = torch.sum(weights_t * residuals_t**2)

            n_points, n_params = len(Y_t), X_t.shape[1]
            sigma_sq_hat = wssr / (n_points - n_params)
            leverage_first = P_t[0, 0]

            if 1 - leverage_first <= 1e-9:
                return False

            studentized_residual_first = residuals_t[0] / torch.sqrt(sigma_sq_hat * (1 - leverage_first) / weights_t[0])
            degrees_of_freedom = n_points - n_params

            chi2_critical = chi2.ppf(1 - significance_level, df=degrees_of_freedom)
            z_critical = norm.ppf(1 - significance_level / 2)

            return wssr.item() < chi2_critical and torch.abs(studentized_residual_first).item() < z_critical

        except torch.linalg.LinAlgError:
            return False

    def _get_optimal_threshold_pytorch(
        self,
        independent_peaks: np.ndarray,
        significance_level: float,
        starting_quantile: float
    ) -> float:
        """
        Identifies the optimal threshold using a binary search strategy with PyTorch.
        Computes MRL statistics only from the starting_quantile onwards.
        """
        # --- Initial Setup ---
        data_t = torch.from_numpy(self.data).to(self.device, dtype=torch.float32)
        all_thresholds_t = torch.from_numpy(independent_peaks).to(self.device, dtype=torch.float32)

        # 1. Find the starting threshold value based on the quantile
        # Note: Assumes independent_peaks (and thus all_thresholds_t) is sorted ascending.
        if starting_quantile > 0.0:
            start_val = torch.quantile(all_thresholds_t, starting_quantile)
            # Find the index to start computation from
            start_idx = torch.searchsorted(all_thresholds_t, start_val, side='left').item()
        else:
            start_idx = 0

        print(f"\n[PyTorch Backend] Starting computation from the {starting_quantile*100:.0f}th percentile of peaks.")
        print(f"Computing MRL statistics for up to {len(all_thresholds_t) - start_idx} candidate thresholds...")

        # --- Computation from Starting Quantile ---
        mrl_means, mrl_vars, mrl_counts, valid_thresholds_list = [], [], [], []
        
        # 2. Compute MRL stats ONLY for thresholds >= the starting quantile value
        for u in all_thresholds_t[start_idx:]: # <-- This is the key change
            excesses_t = data_t[data_t > u]
            n_excesses = excesses_t.shape[0]
            
            # If we find a threshold with < 10 excesses, all subsequent ones
            # (which are higher) will also have < 10, so we can stop.
            if n_excesses < 10:
                break 
                
            valid_thresholds_list.append(u)
            mrl_means.append(torch.mean(excesses_t - u))
            mrl_vars.append(torch.var(excesses_t, unbiased=True))
            mrl_counts.append(n_excesses)

        if not valid_thresholds_list:
            print("[PyTorch Backend] No thresholds with sufficient data found after starting quantile.")
            return np.nan

        # Stack the *filtered* and *computed* lists into tensors
        valid_thresholds_t = torch.stack(valid_thresholds_list)
        mrl_means_t = torch.stack(mrl_means)
        mrl_vars_t = torch.stack(mrl_vars)
        mrl_counts_t = torch.tensor(mrl_counts, device=self.device, dtype=torch.float32)
        print(f"Computation complete. Found {len(valid_thresholds_t)} valid candidate thresholds.")

        # --- Binary Search ---
        # 3. Commence the search on the *new* list.
        #    The search range is now the *entire* new list (index 0 to end),
        #    as it already starts from the desired quantile.
        print(f"Starting binary search on {len(valid_thresholds_t)} computed thresholds...")
        low, high = 0, len(valid_thresholds_t) - 1 # <-- Key change
        optimal_threshold = np.nan

        while low <= high:
            mid = low + (high - low) // 2
            
            # Check if the slice from 'mid' onwards has enough data points
            if len(valid_thresholds_t[mid:]) < 10:
                high = mid - 1
                continue

            is_valid = self._test_threshold_slice_pytorch(
                u_slice=valid_thresholds_t[mid:],
                mean_excesses_slice=mrl_means_t[mid:],
                variances_slice=mrl_vars_t[mid:],
                n_excesses_slice=mrl_counts_t[mid:],
                significance_level=significance_level
            )

            if is_valid:
                optimal_threshold = valid_thresholds_t[mid].item()
                high = mid - 1  # Success, try for an even lower threshold (earlier in the list)
            else:
                low = mid + 1  # Failure, need a higher threshold (later in the list)

        # --- Return Result ---
        if not np.isnan(optimal_threshold):
            print(f"\n[PyTorch Backend] Found optimal threshold: {optimal_threshold:.2f}.")
            return optimal_threshold
        else:
            print("\n[PyTorch Backend] No optimal threshold found that satisfies all criteria.")
            return np.nan

    def _plot_results(
        self,
        found_threshold: float,
        independent_peaks: np.ndarray,
        true_threshold: Optional[float] = None
    ):
        """Generates and displays plots to visualize the data and results."""
        print("\nGenerating plots...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100)

        # # Plot 1: Time Series Data with Peaks and Thresholds
        # ax1.plot(self.data, label='Time Series Data', color='lightblue', zorder=1)
        # peak_indices = np.where(np.isin(self.data, independent_peaks))[0]
        # ax1.scatter(peak_indices, self.data[peak_indices], color='orange', s=15, label='Independent Peaks', zorder=2)
        # if true_threshold is not None:
        #     ax1.axhline(y=true_threshold, color='red', linestyle='--', label=f'True Threshold ({true_threshold:.2f})', zorder=3)
        # if not np.isnan(found_threshold):
        #     ax1.axhline(y=found_threshold, color='green', linestyle='-', label=f'Found Threshold ({found_threshold:.2f})', zorder=4)
        # ax1.set_title('Time Series Data with Thresholds')
        # ax1.set_ylabel('Value')
        # ax1.legend()
        # ax1.set_xlim(0, len(self.data))

        # Plot 2: Mean Residual Life Plot
        # mrl_thresholds = independent_peaks[independent_peaks < np.max(self.data) * 0.98]
        plot_start = np.quantile(independent_peaks, 0.1) 
        mrl_thresholds = independent_peaks[independent_peaks > plot_start]
        # no need to plot the whole thing 
        mean_excesses = [np.mean(self.data[self.data > u] - u) for u in mrl_thresholds if len(self.data[self.data > u]) > 0]
        valid_mrl_thresholds = [u for u in mrl_thresholds if len(self.data[self.data > u]) > 0]
        ax2.plot(valid_mrl_thresholds, mean_excesses, 'o-', markersize=4, label='Mean Excess')
        if true_threshold is not None:
            ax2.axvline(x=true_threshold, color='red', linestyle='--', label='True Threshold')
        if not np.isnan(found_threshold):
            ax2.axvline(x=found_threshold, color='green', linestyle='-', label='Found Threshold')
        ax2.set_title('Mean Residual Life Plot (MRLP)')
        ax2.set_xlabel('Threshold (u)')
        ax2.set_ylabel('Mean Excess E[X-u | X>u]')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def determine_threshold(
        self,
        min_distance_days: int,
        significance_level: float = 0.05,
        starting_quantile: float = 0.0,
        backend: str = 'pytorch',
        plot: bool = False,
        true_threshold: Optional[float] = None
    ) -> float:
        """
        Determines the optimal threshold by running the full analysis pipeline.

        Args:
            min_distance_days (int): The minimum number of days between peaks.
            significance_level (float, optional): Significance level for tests. Defaults to 0.05.
            starting_quantile (float, optional): Quantile of peaks to start search from. Defaults to 0.0.
            backend (str, optional): 'numpy' or 'pytorch'. Defaults to 'numpy'.
            plot (bool, optional): If True, generates and shows plots. Defaults to False.
            true_threshold (float, optional): The known true threshold for plotting. Defaults to None.

        Returns:
            float: The determined optimal threshold, or a fallback proxy if not found.
        """
        # independent_peaks = self._extract_independent_peaks(min_distance_days)
        # independent_peaks = self._extract_spatial_peaks_scipy(
        #     self.coords, self.data, radius = 0.01
        # )
        independent_peaks = np.sort(self.data)
        start_time = time.time()
        optimal_threshold = np.nan

        if backend.lower() == 'pytorch':
            if self.pytorch_available:
                optimal_threshold = self._get_optimal_threshold_pytorch(independent_peaks, significance_level, starting_quantile)
            else:
                print("PyTorch backend requested but not available. Falling back to NumPy.")
                optimal_threshold = self._get_optimal_threshold_numpy(independent_peaks, significance_level, starting_quantile)
        elif backend.lower() == 'numpy':
            optimal_threshold = self._get_optimal_threshold_numpy(independent_peaks, significance_level, starting_quantile)
        else:
            raise ValueError("Invalid backend. Choose 'numpy' or 'pytorch'.")

        end_time = time.time()
        print(f"\n--- Results ---")
        print(f"Execution time: {end_time - start_time:.4f} seconds")

        if not np.isnan(optimal_threshold):
            quantile = np.mean(self.data <= optimal_threshold)
            print(f"The algorithm identified an optimal threshold of: {optimal_threshold:.2f}")
            print(f"This corresponds to the {quantile:.2%} quantile of the original data.")
            if true_threshold is not None:
                error = abs(optimal_threshold - true_threshold)
                print(f"Absolute error from true threshold: {error:.2f}")
        else:
            print("The algorithm could not determine an optimal threshold.")
            backup_quantile = 0.96
            optimal_threshold = np.quantile(independent_peaks, backup_quantile)
            print(f"Returning the {int(100*backup_quantile):.2f}th quantile {optimal_threshold:.2f} as a fallback proxy.")

        if plot:
            self._plot_results(optimal_threshold, independent_peaks, true_threshold)

        return optimal_threshold


if __name__ == "__main__":
    # --- 1. Generate Synthetic Data for Demonstration ---
    n_points = 15000
    bulk_data = np.random.normal(loc=5, scale=3, size=n_points)
    bulk_data[bulk_data < 0] = 0  # Ensure non-negativity

    # Define a true threshold and generate a Pareto tail above it
    true_quantile = np.random.uniform(0.9, 0.99)
    true_threshold_value = np.quantile(bulk_data, true_quantile)
    shape_param_xi, scale_param_sigma = 0.15, 5.0
    
    tail_indices = np.where(bulk_data > true_threshold_value)[0]
    tail_data = genpareto.rvs(c=shape_param_xi, loc=true_threshold_value, scale=scale_param_sigma, size=len(tail_indices))
    
    synthetic_data = bulk_data.copy()
    synthetic_data[tail_indices] = tail_data

    print("--- Synthetic Data Generation ---")
    print(f"Generated {n_points} data points.")
    print(f"True Threshold Quantile: {true_quantile:.2%}, Value: {true_threshold_value:.2f}")
    print(f"Number of points in tail: {len(tail_indices)}")
    print("-" * 30)

    # --- 2. Initialize and Run the Analyzer ---
    analyzer = ThresholdAnalyzer(synthetic_data)

    # Determine the threshold using desired parameters.
    # Switch the backend to 'pytorch' to test the PyTorch implementation.
    found_threshold = analyzer.determine_threshold(
        min_distance_days=3,
        significance_level=0.05,
        starting_quantile=0.60,
        backend='numpy',  # or 'pytorch'
        plot=True,
        true_threshold=true_threshold_value
    )
    
    print("-" * 30)