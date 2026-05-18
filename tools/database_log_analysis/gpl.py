import pandas as pd
import numpy as np
from typing import Tuple, Dict
from .core import AnalysisModule

class GplAnalysis(AnalysisModule):
    """
    Analysis module for the GPL (Global Placer) tool.
    Provides access to Nesterov placement metrics, gradients, and bin grids
    as Pandas DataFrames along with column descriptions.
    """

    @property
    def iteration_scalars(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Iteration-level scalar values.
        """
        df = self.db.get_table("gpl_iteration_scalars")
        desc = {
            "Iter": "The Nesterov iteration number",
            "StepLength": "The step length used for the iteration",
            "DensityPenalty": "The density penalty parameter",
            "WlCoefX": "Wirelength coefficient for X gradients",
            "WlCoefY": "Wirelength coefficient for Y gradients",
            "BaseWlCoef": "Base wirelength coefficient",
            "SumOverflow": "Total unscaled density overflow"
        }
        return df, desc

    @property
    def bin_grid(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Bin grid state at various iterations.
        """
        df = self.db.get_table("gpl_bin_grid")
        desc = {
            "Iter": "The Nesterov iteration number",
            "BinIdx": "Flat index of the bin",
            "ElectroFieldX": "Electrostatic field X-component",
            "ElectroFieldY": "Electrostatic field Y-component",
            "Density": "Bin density (utilization)"
        }
        return df, desc

    @property
    def cell_static_info(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Static properties of cells (logged once).
        """
        df = self.db.get_table("gpl_cell_static_info")
        desc = {
            "CellId": "The index of the cell in the placer",
            "Width": "Cell width in dbu",
            "Height": "Cell height in dbu",
            "IsMacro": "1 if the cell is a macro, 0 otherwise",
            "IsLocked": "1 if the cell is locked/fixed, 0 otherwise"
        }
        return df, desc

    @property
    def cell_dense_gradients(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Dense gradient components for cells across iterations.
        """
        df = self.db.get_table("gpl_cell_dense_gradients")
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "PosX": "Current X position of the cell",
            "PosY": "Current Y position of the cell",
            "WlX": "Wirelength gradient X-component",
            "WlY": "Wirelength gradient Y-component"
        }
        return df, desc

    @property
    def cell_timing_gradients(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Sparse timing gradient components.
        """
        try:
            df = self.db.get_table("gpl_cell_timing_gradients")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "TimX": "Timing gradient X-component",
            "TimY": "Timing gradient Y-component"
        }
        return df, desc

    @property
    def cell_routability_gradients(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Sparse routability gradient components.
        """
        try:
            df = self.db.get_table("gpl_cell_routability_gradients")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "RtX": "Routability gradient X-component",
            "RtY": "Routability gradient Y-component"
        }
        return df, desc

    @property
    def cell_derived_gradients(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Precomputed gradient metrics (magnitudes, dot products, opposition).
        """
        try:
            df = self.db.get_table("gpl_derived_gradients")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "mag_wl": "Magnitude of the Wirelength gradient",
            "mag_tim": "Magnitude of the Timing gradient",
            "dot_wl_tim": "Dot product of WL and Timing gradients",
            "opposition": "Measure of conflict between WL and Timing forces"
        }
        return df, desc


    @property
    def path_slacks(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Slack for timing paths.
        """
        try:
            df = self.db.get_table("gpl_path_slacks")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "PathId": "Unique ID for the timing path",
            "Slack": "Slack value of the path",
            "Iter": "The Nesterov iteration number"
        }
        return df, desc

    @property
    def path_cells(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Cell memberships for timing paths.
        """
        try:
            df = self.db.get_table("gpl_path_cells")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "PathId": "Unique ID for the timing path",
            "CellId": "The index of the cell participating in the path",
            "Iter": "The Nesterov iteration number"
        }
        return df, desc

    def get_max_overflow_by_iteration(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Returns the overflow trend during placement.
        """
        df = self.db.query("SELECT Iter, SumOverflow FROM [gpl_iteration_scalars] ORDER BY Iter ASC")
        desc = {
            "Iter": "The Nesterov iteration number",
            "SumOverflow": "Total unscaled density overflow at this iteration"
        }
        return df, desc

    def get_cell_movements(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Computes the displacement of each cell between iterations.
        Calculates DeltaX, DeltaY, and Euclidean Distance moved using NumPy vectorization.
        """
        df, _ = self.cell_dense_gradients
        if df.empty:
            return df, {}
            
        # Ensure we work with a copy and sorted data
        df = df.sort_values(by=["CellId", "Iter"])
        
        # Calculate diffs using numpy to avoid groupby overhead
        mask = df["CellId"].values[1:] == df["CellId"].values[:-1]
        
        pos_x = df["PosX"].values
        pos_y = df["PosY"].values
        
        delta_x = np.zeros_like(pos_x)
        delta_y = np.zeros_like(pos_y)
        
        delta_x[1:][mask] = pos_x[1:][mask] - pos_x[:-1][mask]
        delta_y[1:][mask] = pos_y[1:][mask] - pos_y[:-1][mask]
        
        df["DeltaX"] = delta_x
        df["DeltaY"] = delta_y
        df["Distance"] = np.sqrt(delta_x**2 + delta_y**2)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "PosX": "Current X position of the cell",
            "PosY": "Current Y position of the cell",
            "DeltaX": "Change in X position since previous logged iteration",
            "DeltaY": "Change in Y position since previous logged iteration",
            "Distance": "Euclidean distance moved since previous logged iteration"
        }
        
        return df[["Iter", "CellId", "PosX", "PosY", "DeltaX", "DeltaY", "Distance"]], desc

    def _get_gradient_vectors_helper(self, df: pd.DataFrame, x_col: str, y_col: str, prefix: str) -> pd.DataFrame:
        """Helper to compute magnitude and unit vector components using NumPy."""
        if df.empty:
            return df
        
        x = df[x_col].values
        y = df[y_col].values
        mag = np.sqrt(x**2 + y**2)
        
        # Create result DataFrame
        res = df.copy()
        res[f"{prefix}Mag"] = mag
        
        # Vectorized unit vector calculation
        mask = mag > 0
        unit_x = np.zeros_like(x)
        unit_y = np.zeros_like(y)
        unit_x[mask] = x[mask] / mag[mask]
        unit_y[mask] = y[mask] / mag[mask]
        
        res[f"{prefix}UnitX"] = unit_x
        res[f"{prefix}UnitY"] = unit_y
        return res

    def get_wl_gradient_vectors(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Wirelength gradients."""
        df, _ = self.cell_dense_gradients
        if df.empty: return df, {}
        res = self._get_gradient_vectors_helper(df.copy(), "WlX", "WlY", "Wl")
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "WlMag": "Magnitude of the Wirelength gradient vector",
            "WlUnitX": "X-component of the unit Wirelength gradient vector",
            "WlUnitY": "Y-component of the unit Wirelength gradient vector"
        }
        return res, desc

    def get_tim_gradient_vectors(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Timing gradients."""
        df, _ = self.cell_timing_gradients
        if df.empty: return df, {}
        res = self._get_gradient_vectors_helper(df.copy(), "TimX", "TimY", "Tim")
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "TimMag": "Magnitude of the Timing gradient vector",
            "TimUnitX": "X-component of the unit Timing gradient vector",
            "TimUnitY": "Y-component of the unit Timing gradient vector"
        }
        return res, desc

    def get_estimated_density_forces(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Estimates the density force applied to each cell using a 
        Summed Area Table (Integral Image) approach for O(1) force calculation per cell.
        This is thousands of times faster than the iterative overlap method.
        """
        # 1. Get all necessary data
        cells_df, _ = self.cell_dense_gradients
        if cells_df.empty:
            return pd.DataFrame(), {}
        
        static_df, _ = self.cell_static_info
        bins_df, _ = self.bin_grid
        scalars_df, _ = self.iteration_scalars
        
        # Metadata
        try:
            lx = float(self.db.get_metadata(f"region_{region_name}_lx")[0])
            ly = float(self.db.get_metadata(f"region_{region_name}_ly")[0])
            bin_size_x = float(self.db.get_metadata(f"region_{region_name}_binSizeX")[0])
            bin_size_y = float(self.db.get_metadata(f"region_{region_name}_binSizeY")[0])
            bin_cnt_x = int(self.db.get_metadata(f"region_{region_name}_binCntX")[0])
            bin_cnt_y = int(self.db.get_metadata(f"region_{region_name}_binCntY")[0])
        except (IndexError, ValueError):
            raise ValueError(f"Missing electrostatic bin grid metadata for region '{region_name}'.")

        # 2. Build the Integral Image (Prefix Sum) of the Electrostatic Field
        iters = np.sort(bins_df["Iter"].unique())
        num_iters = len(iters)
        iter_to_idx = {it: i for i, it in enumerate(iters)}
        
        field_x_grid = np.zeros((num_iters, bin_cnt_y, bin_cnt_x), dtype=np.float32)
        field_y_grid = np.zeros((num_iters, bin_cnt_y, bin_cnt_x), dtype=np.float32)
        
        iter_idxs = np.array([iter_to_idx[it] for it in bins_df["Iter"]])
        bin_idxs = bins_df["BinIdx"].values
        bx_idxs = bin_idxs % bin_cnt_x
        by_idxs = bin_idxs // bin_cnt_x
        
        field_x_grid[iter_idxs, by_idxs, bx_idxs] = bins_df["ElectroFieldX"].values
        field_y_grid[iter_idxs, by_idxs, bx_idxs] = bins_df["ElectroFieldY"].values
        
        sat_x = np.zeros((num_iters, bin_cnt_y + 1, bin_cnt_x + 1), dtype=np.float64)
        sat_y = np.zeros((num_iters, bin_cnt_y + 1, bin_cnt_x + 1), dtype=np.float64)
        
        bin_area = bin_size_x * bin_size_y
        sat_x[:, 1:, 1:] = np.cumsum(np.cumsum(field_x_grid, axis=1), axis=2) * bin_area
        sat_y[:, 1:, 1:] = np.cumsum(np.cumsum(field_y_grid, axis=1), axis=2) * bin_area
        
        # 3. Vectorized Force Calculation
        cell_info = pd.merge(cells_df, static_df[["CellId", "Width", "Height"]], on="CellId")
        cell_info = pd.merge(cell_info, scalars_df[["Iter", "DensityPenalty"]], on="Iter")
        
        pos_x = cell_info["PosX"].values
        pos_y = cell_info["PosY"].values
        width = cell_info["Width"].values
        height = cell_info["Height"].values
        penalty = cell_info["DensityPenalty"].values
        c_iter_idxs = np.array([iter_to_idx[it] for it in cell_info["Iter"]])
        
        c_lx = (pos_x - width/2 - lx) / bin_size_x
        c_ly = (pos_y - height/2 - ly) / bin_size_y
        c_ux = (pos_x + width/2 - lx) / bin_size_x
        c_uy = (pos_y + height/2 - ly) / bin_size_y
        
        c_lx = np.clip(c_lx, 0, bin_cnt_x)
        c_ly = np.clip(c_ly, 0, bin_cnt_y)
        c_ux = np.clip(c_ux, 0, bin_cnt_x)
        c_uy = np.clip(c_uy, 0, bin_cnt_y)
        
        def interpolate_sat(sat, iter_idx, x, y):
            ix = np.floor(x).astype(int)
            iy = np.floor(y).astype(int)
            fx = x - ix
            fy = y - iy
            # Clamp ix, iy for safety
            ix = np.clip(ix, 0, bin_cnt_x-1)
            iy = np.clip(iy, 0, bin_cnt_y-1)
            
            v00 = sat[iter_idx, iy, ix]
            v10 = sat[iter_idx, iy, ix+1]
            v01 = sat[iter_idx, iy+1, ix]
            v11 = sat[iter_idx, iy+1, ix+1]
            return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11

        i_lx_ly_x = interpolate_sat(sat_x, c_iter_idxs, c_lx, c_ly)
        i_ux_ly_x = interpolate_sat(sat_x, c_iter_idxs, c_ux, c_ly)
        i_lx_uy_x = interpolate_sat(sat_x, c_iter_idxs, c_lx, c_uy)
        i_ux_uy_x = interpolate_sat(sat_x, c_iter_idxs, c_ux, c_uy)
        force_x = (i_ux_uy_x - i_lx_uy_x - i_ux_ly_x + i_lx_ly_x) * penalty
        
        i_lx_ly_y = interpolate_sat(sat_y, c_iter_idxs, c_lx, c_ly)
        i_ux_ly_y = interpolate_sat(sat_y, c_iter_idxs, c_ux, c_ly)
        i_lx_uy_y = interpolate_sat(sat_y, c_iter_idxs, c_lx, c_uy)
        i_ux_uy_y = interpolate_sat(sat_y, c_iter_idxs, c_ux, c_uy)
        force_y = (i_ux_uy_y - i_lx_uy_y - i_ux_ly_y + i_lx_ly_y) * penalty
        
        res = cell_info[["Iter", "CellId"]].copy()
        res["EstDensityForceX"] = force_x.astype(np.float32)
        res["EstDensityForceY"] = force_y.astype(np.float32)
        res["EstDensityForceMag"] = np.sqrt(force_x**2 + force_y**2).astype(np.float32)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "EstDensityForceX": "Estimated density gradient force (X) applied to cell",
            "EstDensityForceY": "Estimated density gradient force (Y) applied to cell",
            "EstDensityForceMag": "Magnitude of the estimated density force vector"
        }
        return res, desc

    def get_density_force_vectors(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Estimated Density Forces."""
        res, _ = self.get_estimated_density_forces(region_name)
        if res.empty: return res, {}
        mag = res["EstDensityForceMag"].values
        mask = mag > 0
        res["EstDensityForceUnitX"] = np.where(mask, res["EstDensityForceX"] / mag, 0.0)
        res["EstDensityForceUnitY"] = np.where(mask, res["EstDensityForceY"] / mag, 0.0)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "EstDensityForceMag": "Magnitude of the estimated density force vector",
            "EstDensityForceUnitX": "X-component of the unit estimated density force vector",
            "EstDensityForceUnitY": "Y-component of the unit estimated density force vector"
        }
        return res, desc

    def get_bin_analytics(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Aggregates bin density per iteration to find max, mean, and std density.
        """
        df, _ = self.bin_grid
        if df.empty: return df, {}
        agg_df = df.groupby("Iter")["Density"].agg(
            MaxDensity='max', MeanDensity='mean', StdDensity='std'
        ).reset_index()
        desc = {
            "Iter": "The Nesterov iteration number",
            "MaxDensity": "Maximum bin density at this iteration",
            "MeanDensity": "Average bin density across all bins",
            "StdDensity": "Standard deviation of bin density"
        }
        return agg_df, desc

    def get_cell_bin_mapping(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Calculates which bin index each cell falls into at each iteration.
        """
        df, _ = self.cell_dense_gradients
        if df.empty: return df, {}
        try:
            lx = float(self.db.get_metadata(f"region_{region_name}_lx")[0])
            ly = float(self.db.get_metadata(f"region_{region_name}_ly")[0])
            bin_size_x = float(self.db.get_metadata(f"region_{region_name}_binSizeX")[0])
            bin_size_y = float(self.db.get_metadata(f"region_{region_name}_binSizeY")[0])
            bin_cnt_x = int(self.db.get_metadata(f"region_{region_name}_binCntX")[0])
        except:
            raise ValueError(f"Missing electrostatic bin grid metadata for region '{region_name}'.")

        idx_x = np.clip(np.floor((df["PosX"].values - lx) / bin_size_x).astype(int), 0, bin_cnt_x - 1)
        idx_y = np.maximum(np.floor((df["PosY"].values - ly) / bin_size_y).astype(int), 0)
        df["BinIdx"] = idx_y * bin_cnt_x + idx_x
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "BinIdx": "The calculated flat index of the electrostatic bin"
        }
        return df[["Iter", "CellId", "BinIdx"]], desc

    def get_path_slack_trends(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Computes the minimum, maximum, and mean slack across all logged paths per iteration.
        """
        df, _ = self.path_slacks
        if df.empty: return df, {}
        agg_df = df.groupby("Iter")["Slack"].agg(
            MinSlack='min', MaxSlack='max', MeanSlack='mean'
        ).reset_index()
        desc = {
            "Iter": "The Nesterov iteration number",
            "MinSlack": "The worst (minimum) slack among all logged paths",
            "MaxSlack": "The best (maximum) slack among all logged paths",
            "MeanSlack": "The average slack among all logged paths"
        }
        return agg_df, desc
