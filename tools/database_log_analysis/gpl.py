import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

try:
    from .core import AnalysisModule
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from database_log_analysis.core import AnalysisModule

class GplAnalysis(AnalysisModule):
    """
    Analysis module for the GPL (Global Placer) tool.
    Provides access to Nesterov placement metrics, gradients, and bin grids
    as Pandas DataFrames along with column descriptions.
    
    NOTE ON SPARSITY:
    Routability and timing gradients are sparse by design. A missing record in
    these tables for a given (Iteration, CellId) implicitly means the gradient
    force was exactly zero for that step.
    """

    def _build_filter_query(self, table_name: str, iter_range: Optional[Tuple[int, int]] = None, 
                            cell_range: Optional[Tuple[int, int]] = None, 
                            cell_ids: Optional[List[int]] = None) -> str:
        """Helper to build a SQL query with chunking/filtering."""
        query = f"SELECT * FROM [{table_name}]"
        conditions = []
        
        if iter_range:
            conditions.append(f"Iter >= {iter_range[0]} AND Iter <= {iter_range[1]}")
            
        if cell_range:
            conditions.append(f"CellId >= {cell_range[0]} AND CellId <= {cell_range[1]}")
            
        if cell_ids is not None:
            if len(cell_ids) == 0:
                return f"SELECT * FROM [{table_name}] WHERE 1=0"
            ids_str = ",".join(map(str, cell_ids))
            conditions.append(f"CellId IN ({ids_str})")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        return query

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

    def get_bin_grid(self, iter_range: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Bin grid state at various iterations (Supports piece-wise loading).
        """
        sql = self._build_filter_query("gpl_bin_grid", iter_range=iter_range)
        df = self.db.query(sql)
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

    def get_cell_dense_gradients(self, iter_range: Optional[Tuple[int, int]] = None,
                                 cell_range: Optional[Tuple[int, int]] = None,
                                 cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Dense gradient components for cells across iterations (Supports piece-wise loading).
        """
        sql = self._build_filter_query("gpl_cell_dense_gradients", iter_range, cell_range, cell_ids)
        df = self.db.query(sql)
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "PosX": "Current X position of the cell",
            "PosY": "Current Y position of the cell",
            "WlX": "Wirelength gradient X-component",
            "WlY": "Wirelength gradient Y-component"
        }
        return df, desc

    def get_cell_timing_gradients(self, iter_range: Optional[Tuple[int, int]] = None,
                                  cell_range: Optional[Tuple[int, int]] = None,
                                  cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Sparse timing gradient components (Supports piece-wise loading).
        NOTE: Missing records imply 0.0 force for that iteration/cell.
        """
        try:
            sql = self._build_filter_query("gpl_cell_timing_gradients", iter_range, cell_range, cell_ids)
            df = self.db.query(sql)
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "TimX": "Timing gradient X-component",
            "TimY": "Timing gradient Y-component"
        }
        return df, desc

    def get_cell_routability_gradients(self, iter_range: Optional[Tuple[int, int]] = None,
                                       cell_range: Optional[Tuple[int, int]] = None,
                                       cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Sparse routability gradient components (Supports piece-wise loading).
        NOTE: Missing records imply 0.0 force for that iteration/cell.
        """
        try:
            sql = self._build_filter_query("gpl_cell_routability_gradients", iter_range, cell_range, cell_ids)
            df = self.db.query(sql)
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "RtX": "Routability gradient X-component",
            "RtY": "Routability gradient Y-component"
        }
        return df, desc

    def get_cell_derived_gradients(self, iter_range: Optional[Tuple[int, int]] = None,
                                   cell_range: Optional[Tuple[int, int]] = None,
                                   cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Precomputed gradient metrics (magnitudes, dot products, opposition).
        Supports piece-wise loading to prevent OOM on 22M+ rows.
        """
        try:
            sql = self._build_filter_query("gpl_derived_gradients", iter_range, cell_range, cell_ids)
            df = self.db.query(sql)
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
    def path_signature_map(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Map from (PathId, Iter) to stable PhysicalPathId.
        """
        try:
            df = self.db.get_table("path_signature_map")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "PathId": "Original unstable PathId",
            "Iter": "The Nesterov iteration number",
            "PhysicalPathId": "Stable identifier for the physical path"
        }
        return df, desc

    @property
    def path_cells(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Cell memberships for timing paths.
        Each row represents one cell's participation in a timing path at a given iteration.
        PathSeq indicates the cell's position in the path (0 = source, N-1 = sink).
        Slack is the per-stage slack at this cell, providing finer granularity than the overall path slack.
        """
        try:
            df = self.db.get_table("gpl_path_cells")
        except ValueError:
            df = pd.DataFrame()
            
        desc = {
            "PathId": "Unique ID for the timing path",
            "CellId": "The index of the cell participating in the path",
            "Iter": "The Nesterov iteration number",
            "PathSeq": "Position of the cell in the path (0=source, increasing towards sink)",
            "Slack": "Per-stage slack at this cell position (ps)"
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

    def get_cell_movements(self, iter_range: Optional[Tuple[int, int]] = None,
                           cell_range: Optional[Tuple[int, int]] = None,
                           cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Computes the displacement of each cell between iterations.
        Calculates DeltaX, DeltaY, and Euclidean Distance moved using NumPy vectorization.
        Supports piece-wise loading.
        """
        df, _ = self.get_cell_dense_gradients(iter_range, cell_range, cell_ids)
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

    def get_wl_gradient_vectors(self, iter_range: Optional[Tuple[int, int]] = None,
                                cell_range: Optional[Tuple[int, int]] = None,
                                cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Wirelength gradients. Supports piece-wise loading."""
        df, _ = self.get_cell_dense_gradients(iter_range, cell_range, cell_ids)
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

    def get_tim_gradient_vectors(self, iter_range: Optional[Tuple[int, int]] = None,
                                 cell_range: Optional[Tuple[int, int]] = None,
                                 cell_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Timing gradients. Supports piece-wise loading."""
        df, _ = self.get_cell_timing_gradients(iter_range, cell_range, cell_ids)
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

    def get_estimated_density_forces(self, iter_range: Optional[Tuple[int, int]] = None,
                                     cell_range: Optional[Tuple[int, int]] = None,
                                     cell_ids: Optional[List[int]] = None,
                                     region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Estimates the density force applied to each cell using a 
        Summed Area Table (Integral Image) approach for O(1) force calculation per cell.
        This is thousands of times faster than the iterative overlap method.
        Supports piece-wise loading.
        """
        # 1. Get all necessary data (filtered)
        cells_df, _ = self.get_cell_dense_gradients(iter_range, cell_range, cell_ids)
        if cells_df.empty:
            return pd.DataFrame(), {}
        
        static_df, _ = self.cell_static_info
        bins_df, _ = self.get_bin_grid(iter_range)
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

    def get_density_force_vectors(self, iter_range: Optional[Tuple[int, int]] = None,
                                  cell_range: Optional[Tuple[int, int]] = None,
                                  cell_ids: Optional[List[int]] = None,
                                  region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Estimated Density Forces. Supports piece-wise loading."""
        res, _ = self.get_estimated_density_forces(iter_range, cell_range, cell_ids, region_name)
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

    def get_bin_analytics(self, iter_range: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Aggregates bin density per iteration to find max, mean, and std density.
        Supports piece-wise loading.
        """
        df, _ = self.get_bin_grid(iter_range)
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

    def get_cell_bin_mapping(self, iter_range: Optional[Tuple[int, int]] = None,
                             cell_range: Optional[Tuple[int, int]] = None,
                             cell_ids: Optional[List[int]] = None,
                             region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Calculates which bin index each cell falls into at each iteration.
        Supports piece-wise loading.
        """
        df, _ = self.get_cell_dense_gradients(iter_range, cell_range, cell_ids)
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

    def get_worst_paths_history(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retrieves the full history of slacks for paths that were among the worst 'top_n'
        at ANY iteration. This is a workflow method that combines slacks and stable signatures.
        """
        # Get path slacks
        df_slacks, _ = self.path_slacks
        # Get signature map (PhysicalPathId)
        df_sig, _ = self.path_signature_map

        if df_slacks.empty:
            return pd.DataFrame()
        
        # If signatures are missing, we can't do stable tracking, but we'll return raw if possible
        if df_sig.empty:
            print("Warning: path_signature_map is missing. Stable path tracking is disabled.")
            # Fallback to unstable PathId
            df = df_slacks.copy()
            df["PhysicalPathId"] = df["PathId"]
        else:
            # Merge slacks with signatures
            df = pd.merge(df_slacks, df_sig, on=["PathId", "Iter"])

        # Identify the PhysicalPathIds that were in the bottom N at some iteration
        def _get_worst_n_ids(group):
            return group.nsmallest(top_n, "Slack")["PhysicalPathId"]

        worst_phys_ids = df.groupby("Iter").apply(_get_worst_n_ids, include_groups=False).unique()
        
        # Extract full history for these specific physical paths
        history = df[df["PhysicalPathId"].isin(worst_phys_ids)].copy()
        return history.sort_values(["PhysicalPathId", "Iter"])

    def get_path_force_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Extracts detailed force/gradient data for every cell participating in the
        worst 'top_n' paths across all iterations.
        """
        # 1. Get the history of worst paths to know which (PathId, Iter) to look at
        history = self.get_worst_paths_history(top_n=top_n)
        if history.empty:
            return pd.DataFrame()
            
        # 2. Get the cells for these paths at these iterations
        all_path_cells, _ = self.path_cells
        if all_path_cells.empty:
            return pd.DataFrame()
            
        # Merge with history to get PhysicalPathId and filter for only the paths of interest
        merged_cells = pd.merge(all_path_cells, history[['PathId', 'Iter', 'PhysicalPathId', 'Slack']], 
                               on=['PathId', 'Iter'], suffixes=('_cell', '_path'))
        
        # 3. Get Gradient Data
        cell_ids = merged_cells['CellId'].unique().tolist()
        # IMPORTANT: Timing gradients are logged EVERY iteration, while paths are sparse.
        # We want to see the timing force at every iteration, not just when paths are logged.
        # So we fetch the full iteration range for these cells.
        iter_min = history['Iter'].min()
        iter_max = history['Iter'].max()
        iter_range = (iter_min, iter_max)
        
        wl_df, _ = self.get_wl_gradient_vectors(iter_range=iter_range, cell_ids=cell_ids)
        tim_df, _ = self.get_tim_gradient_vectors(iter_range=iter_range, cell_ids=cell_ids)
        dens_df, _ = self.get_density_force_vectors(iter_range=iter_range, cell_ids=cell_ids)
        
        # 4. Merge all data
        # We start with the cross product of (Iter x CellId) from the gradients to fill gaps
        # OR better: merge gradients together first, then left join the path info.
        # Since gradients are the "dense" part in time (every iteration).
        
        # Merge WL and Timing first (Timing is the core interest)
        df = pd.merge(wl_df, tim_df, on=['Iter', 'CellId'], how='outer')
        df = pd.merge(df, dens_df, on=['Iter', 'CellId'], how='left')
        
        # Now join the path membership info. 
        # Note: A cell might belong to DIFFERENT paths. We'll join such that we know 
        # which path(s) it belongs to. Since path logging is sparse, we "forward fill" 
        # the path membership for the analysis? 
        # No, let's keep it simple: join on Iter/CellId. Path info will be sparse.
        df = pd.merge(df, merged_cells, on=['Iter', 'CellId'], how='left')
        
        # Fill NaNs for sparse gradients
        for col in ['WlMag', 'TimMag', 'EstDensityForceMag', 
                    'WlUnitX', 'WlUnitY', 'TimUnitX', 'TimUnitY', 
                    'EstDensityForceUnitX', 'EstDensityForceUnitY']:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                
        # 5. Alignment metrics
        df['Alignment_Tim_WL'] = df['TimUnitX'] * df['WlUnitX'] + df['TimUnitY'] * df['WlUnitY']
        df['Alignment_Tim_Dens'] = df['TimUnitX'] * df['EstDensityForceUnitX'] + \
                                   df['TimUnitY'] * df['EstDensityForceUnitY']
                                   
        other_x = df['WlUnitX'] * df['WlMag'] + df['EstDensityForceUnitX'] * df['EstDensityForceMag']
        other_y = df['WlUnitY'] * df['WlMag'] + df['EstDensityForceUnitY'] * df['EstDensityForceMag']
        other_mag = np.sqrt(other_x**2 + other_y**2)
        
        mask = other_mag > 0
        df['Opposition_Score'] = 0.0
        df.loc[mask, 'Opposition_Score'] = -(df.loc[mask, 'TimUnitX'] * (other_x[mask]/other_mag[mask]) + \
                                            df.loc[mask, 'TimUnitY'] * (other_y[mask]/other_mag[mask]))
                                            
        return df



    def get_cell_criticality(self, agg_metric: str = "mean") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Aggregates per-cell slack across all paths and iterations to identify
        cells that consistently participate in timing-critical paths.
        Uses the per-cell Slack column from gpl_path_cells, which provides finer granularity
        than path-level slack alone.
        
        Args:
            agg_metric: Aggregation function ('mean', 'min', 'max', 'sum', or 'count').
                        'mean' gives the average cell criticality across all paths.
                        'min' gives the worst slack this cell experienced in any path.
                        'count' gives how many timing paths the cell participates in.
        
        NOTE: Requires the gpl_path_cells table to have the Slack column (new format).
        If the table uses the old format (PathId, CellId, Iter only), returns empty.
        """
        path_cells_df, _ = self.path_cells
        if path_cells_df.empty:
            return pd.DataFrame(), {}

        # Safeguard: check that Slack column exists (new format)
        if "Slack" not in path_cells_df.columns:
            return pd.DataFrame(), {}

        agg_funcs = {
            "mean": "mean",
            "min": "min",
            "max": "max",
            "sum": "sum",
            "count": "count",
        }
        if agg_metric not in agg_funcs:
            raise ValueError(f"Unknown agg_metric '{agg_metric}'. Choose from: {list(agg_funcs.keys())}")

        # Aggregate per-cell slack across paths and iterations
        if agg_metric == "count":
            # Count how many distinct (PathId, Iter) pairs a cell participates in
            crit = (
                path_cells_df.groupby("CellId")
                .agg(PathCount=pd.NamedAgg(column="PathId", aggfunc="nunique"),
                     AvgSlack=pd.NamedAgg(column="Slack", aggfunc="mean"),
                     MinSlack=pd.NamedAgg(column="Slack", aggfunc="min"))
                .reset_index()
            )
            desc = {
                "CellId": "The index of the cell in the placer",
                "PathCount": "Number of distinct timing paths this cell participates in",
                "AvgSlack": "Average per-stage slack across all path participations (ps)",
                "MinSlack": "Worst (minimum) per-stage slack experienced by this cell (ps)"
            }
        else:
            crit = (
                path_cells_df.groupby("CellId")["Slack"]
                .agg(agg_metric)
                .reset_index()
                .rename(columns={"Slack": f"{agg_metric.capitalize()}Slack"})
            )
            desc = {
                "CellId": "The index of the cell in the placer",
                f"{agg_metric.capitalize()}Slack": f"{agg_metric.capitalize()} per-stage slack across all path participations (ps)"
            }

        # Sort by worst (most negative / smallest) slack and add criticality rank
        slack_col = [c for c in crit.columns if c != "CellId" and "Slack" in c][0]
        crit = crit.sort_values(slack_col)
        crit["CriticalityRank"] = range(1, len(crit) + 1)
        desc["CriticalityRank"] = "Rank of cell criticality (1 = most timing-critical)"

        return crit, desc

    def get_gradient_balance_stats(self, iter_range: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Calculates the ratio and statistics of Timing vs Wirelength gradients.
        Helps identify if timing forces are being drowned out or are too aggressive.
        """
        df, _ = self.get_cell_derived_gradients(iter_range=iter_range)
        if df.empty: return df, {}
        
        # Calculate ratio, handling zeros
        df['ratio_tim_wl'] = np.where(df['mag_wl'] > 0, df['mag_tim'] / df['mag_wl'], 0.0)
        
        agg_df = df.groupby("Iter")['ratio_tim_wl'].agg(
            MeanRatio='mean', MedianRatio='median', MaxRatio='max', StdRatio='std'
        ).reset_index()
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "MeanRatio": "Average ratio of Timing Gradient magnitude to WL Gradient magnitude",
            "MedianRatio": "Median ratio of Timing vs WL magnitude",
            "MaxRatio": "Maximum ratio observed in any cell",
            "StdRatio": "Standard deviation of the force ratio"
        }
        return agg_df, desc
