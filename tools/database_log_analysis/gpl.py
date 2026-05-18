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
            
        # Sort by cell and then by iteration for accurate diffs
        df = df.sort_values(by=["CellId", "Iter"])
        
        # Calculate diffs within each cell's group
        df["DeltaX"] = df.groupby("CellId")["PosX"].diff().fillna(0.0)
        df["DeltaY"] = df.groupby("CellId")["PosY"].diff().fillna(0.0)
        df["Distance"] = np.sqrt(df["DeltaX"]**2 + df["DeltaY"]**2)
        
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
        """Helper to compute magnitude and unit vector components."""
        mag = np.sqrt(df[x_col]**2 + df[y_col]**2)
        df[f"{prefix}Mag"] = mag
        df[f"{prefix}UnitX"] = np.where(mag > 0, df[x_col] / mag, 0.0)
        df[f"{prefix}UnitY"] = np.where(mag > 0, df[y_col] / mag, 0.0)
        return df

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

    def get_density_force_vectors(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Computes magnitude and unit vectors for Estimated Density Forces."""
        res, _ = self.get_estimated_density_forces(region_name)
        if res.empty: return res, {}
        
        # Already has Mag, just add units
        mag = res["EstDensityForceMag"]
        res["EstDensityForceUnitX"] = np.where(mag > 0, res["EstDensityForceX"] / mag, 0.0)
        res["EstDensityForceUnitY"] = np.where(mag > 0, res["EstDensityForceY"] / mag, 0.0)
        
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
        if df.empty:
            return df, {}
            
        # Group by iteration and calculate metrics
        agg_df = df.groupby("Iter")["Density"].agg(
            MaxDensity='max',
            MeanDensity='mean',
            StdDensity='std'
        ).reset_index()
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "MaxDensity": "Maximum bin density at this iteration",
            "MeanDensity": "Average bin density across all bins",
            "StdDensity": "Standard deviation of bin density (measure of evenness)"
        }
        
        return agg_df, desc

    def get_cell_bin_mapping(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Calculates which bin index each cell falls into at each iteration,
        using the electrostatic bin grid metadata. Converts string metadata to numerics.
        """
        df, _ = self.cell_dense_gradients
        if df.empty:
            return df, {}
            
        # Fetch metadata and convert from strings to appropriate numeric types
        try:
            lx = float(self.db.get_metadata(f"region_{region_name}_lx")[0])
            ly = float(self.db.get_metadata(f"region_{region_name}_ly")[0])
            bin_size_x = float(self.db.get_metadata(f"region_{region_name}_binSizeX")[0])
            bin_size_y = float(self.db.get_metadata(f"region_{region_name}_binSizeY")[0])
            bin_cnt_x = int(self.db.get_metadata(f"region_{region_name}_binCntX")[0])
        except IndexError:
            raise ValueError(f"Missing electrostatic bin grid metadata for region '{region_name}'. Have you run GPL with logging enabled?")

        # Calculate bin X and Y indices using numpy
        idx_x = np.floor((df["PosX"] - lx) / bin_size_x).astype(int)
        idx_y = np.floor((df["PosY"] - ly) / bin_size_y).astype(int)
        
        # Clamp indices to ensure they fall within the grid limits
        idx_x = np.clip(idx_x, 0, bin_cnt_x - 1)
        idx_y = np.maximum(idx_y, 0)
        
        # Flat index: y * bin_cnt_x + x
        df["BinIdx"] = idx_y * bin_cnt_x + idx_x
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "PosX": "Current X position of the cell",
            "PosY": "Current Y position of the cell",
            "BinIdx": "The calculated flat index of the electrostatic bin containing the cell"
        }
        
        return df[["Iter", "CellId", "PosX", "PosY", "BinIdx"]], desc

    def get_estimated_density_forces(self, region_name: str = "core") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Estimates the density force applied to each cell by evaluating
        all bins that the cell overlaps with, weighted by overlap area.
        """
        # 1. Get all necessary data
        cells_df, _ = self.cell_dense_gradients # Iter, CellId, PosX, PosY
        if cells_df.empty:
            return pd.DataFrame(), {}
        
        static_df, _ = self.cell_static_info # CellId, Width, Height
        bins_df, _ = self.bin_grid # Iter, BinIdx, ElectroFieldX, ElectroFieldY, Density
        scalars_df, _ = self.iteration_scalars # Iter, DensityPenalty
        
        # Metadata
        lx = float(self.db.get_metadata(f"region_{region_name}_lx")[0])
        ly = float(self.db.get_metadata(f"region_{region_name}_ly")[0])
        bin_size_x = float(self.db.get_metadata(f"region_{region_name}_binSizeX")[0])
        bin_size_y = float(self.db.get_metadata(f"region_{region_name}_binSizeY")[0])
        bin_cnt_x = int(self.db.get_metadata(f"region_{region_name}_binCntX")[0])
        bin_cnt_y = int(self.db.get_metadata(f"region_{region_name}_binCntY")[0])
        
        # Merge cell pos and static info
        cell_info = pd.merge(cells_df, static_df[["CellId", "Width", "Height"]], on="CellId")
        
        # Merge penalty
        cell_info = pd.merge(cell_info, scalars_df[["Iter", "DensityPenalty"]], on="Iter")
        
        # Map bins to a searchable format (Iter, BinIdx) -> Field
        bins_dict = bins_df.set_index(["Iter", "BinIdx"])[["ElectroFieldX", "ElectroFieldY"]].to_dict('index')

        results = []
        
        for _, row in cell_info.iterrows():
            iter_num = row["Iter"]
            cell_id = row["CellId"]
            pos_x = row["PosX"]
            pos_y = row["PosY"]
            width = row["Width"]
            height = row["Height"]
            penalty = row["DensityPenalty"]
            
            # Cell bounding box
            cell_lx = pos_x - width / 2
            cell_ly = pos_y - height / 2
            cell_ux = pos_x + width / 2
            cell_uy = pos_y + height / 2
            
            # Bin indices range
            bin_min_x = np.clip(np.floor((cell_lx - lx) / bin_size_x).astype(int), 0, bin_cnt_x - 1)
            bin_max_x = np.clip(np.floor((cell_ux - lx) / bin_size_x).astype(int), 0, bin_cnt_x - 1)
            bin_min_y = np.clip(np.floor((cell_ly - ly) / bin_size_y).astype(int), 0, bin_cnt_y - 1)
            bin_max_y = np.clip(np.floor((cell_uy - ly) / bin_size_y).astype(int), 0, bin_cnt_y - 1)
            
            force_x = 0.0
            force_y = 0.0
            
            for bx in range(bin_min_x, bin_max_x + 1):
                for by in range(bin_min_y, bin_max_y + 1):
                    # Bin bounding box
                    bin_lx = lx + bx * bin_size_x
                    bin_ly = ly + by * bin_size_y
                    bin_ux = bin_lx + bin_size_x
                    bin_uy = bin_ly + bin_size_y
                    
                    # Intersection
                    inter_lx = max(cell_lx, bin_lx)
                    inter_ly = max(cell_ly, bin_ly)
                    inter_ux = min(cell_ux, bin_ux)
                    inter_uy = min(cell_uy, bin_uy)
                    
                    overlap_area = max(0, inter_ux - inter_lx) * max(0, inter_uy - inter_ly)
                    
                    if overlap_area > 0:
                        bin_idx = by * bin_cnt_x + bx
                        field = bins_dict.get((iter_num, bin_idx))
                        if field:
                            force_x += field["ElectroFieldX"] * overlap_area * penalty
                            force_y += field["ElectroFieldY"] * overlap_area * penalty
            
            results.append({
                "Iter": iter_num,
                "CellId": cell_id,
                "EstDensityForceX": force_x,
                "EstDensityForceY": force_y,
                "EstDensityForceMag": np.sqrt(force_x**2 + force_y**2)
            })
        
        res = pd.DataFrame(results)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "EstDensityForceX": "Estimated density gradient force (X) applied to cell (sum of overlapped bins)",
            "EstDensityForceY": "Estimated density gradient force (Y) applied to cell (sum of overlapped bins)",
            "EstDensityForceMag": "Magnitude of the estimated density force vector"
        }
        
        return res, desc

    def get_path_slack_trends(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Computes the minimum, maximum, and mean slack across all logged paths per iteration.
        """
        df, _ = self.path_slacks
        if df.empty:
            return df, {}
            
        agg_df = df.groupby("Iter")["Slack"].agg(
            MinSlack='min',
            MaxSlack='max',
            MeanSlack='mean'
        ).reset_index()
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "MinSlack": "The worst (minimum) slack among all logged paths",
            "MaxSlack": "The best (maximum) slack among all logged paths",
            "MeanSlack": "The average slack among all logged paths"
        }
        
        return agg_df, desc
