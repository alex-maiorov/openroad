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

    def get_gradient_magnitudes(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: Computes the vector magnitude of the various gradient types 
        (Wirelength, Timing, Routability) using NumPy.
        """
        df_wl, _ = self.cell_dense_gradients
        if df_wl.empty:
            return df_wl, {}
            
        # We start with WL gradients
        res = df_wl[["Iter", "CellId", "WlX", "WlY"]].copy()
        res["WlMag"] = np.sqrt(res["WlX"]**2 + res["WlY"]**2)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "WlMag": "Magnitude of the Wirelength gradient vector"
        }
        
        # Add timing if available
        df_tim, _ = self.cell_timing_gradients
        if not df_tim.empty:
            res = pd.merge(res, df_tim, on=["Iter", "CellId"], how="left")
            res["TimMag"] = np.sqrt(res["TimX"]**2 + res["TimY"]**2)
            desc["TimMag"] = "Magnitude of the Timing gradient vector"
            
        # Add routability if available
        df_rt, _ = self.cell_routability_gradients
        if not df_rt.empty:
            res = pd.merge(res, df_rt, on=["Iter", "CellId"], how="left")
            res["RtMag"] = np.sqrt(res["RtX"]**2 + res["RtY"]**2)
            desc["RtMag"] = "Magnitude of the Routability gradient vector"
            
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
        Derived Data: Estimates the density force applied to each cell by joining the
        cell's calculated bin with the bin's electrostatic field, then scaling by 
        cell area and the density penalty parameter.
        """
        # 1. Get cell bin mappings
        try:
            cells_df, _ = self.get_cell_bin_mapping(region_name)
        except ValueError:
            return pd.DataFrame(), {}
            
        if cells_df.empty:
            return cells_df, {}
            
        # 2. Get bin grid to extract ElectroField
        bins_df, _ = self.bin_grid
        
        # 3. Get cell static info for Area (Width * Height)
        static_df, _ = self.cell_static_info
        static_df["Area"] = static_df["Width"] * static_df["Height"]
        
        # 4. Get density penalty from iteration scalars
        scalars_df, _ = self.iteration_scalars
        scalars_df = scalars_df[["Iter", "DensityPenalty"]]
        
        # Merge cell positions with bin electrostatic fields by joining on Iter and BinIdx
        res = pd.merge(cells_df, bins_df[["Iter", "BinIdx", "ElectroFieldX", "ElectroFieldY", "Density"]], 
                       on=["Iter", "BinIdx"], how="left")
                       
        # Merge in cell area
        res = pd.merge(res, static_df[["CellId", "Area", "IsMacro"]], on="CellId", how="left")
        
        # Merge in density penalty for the iteration
        res = pd.merge(res, scalars_df, on="Iter", how="left")
        
        # Calculate estimated density force: ElectroField * Area * DensityPenalty
        # This is a powerful metric that isn't logged directly but dictates the RePlAce expansion force.
        res["EstDensityForceX"] = res["ElectroFieldX"] * res["Area"] * res["DensityPenalty"]
        res["EstDensityForceY"] = res["ElectroFieldY"] * res["Area"] * res["DensityPenalty"]
        res["EstDensityForceMag"] = np.sqrt(res["EstDensityForceX"]**2 + res["EstDensityForceY"]**2)
        
        desc = {
            "Iter": "The Nesterov iteration number",
            "CellId": "The index of the cell in the placer",
            "BinIdx": "The bin the cell falls into",
            "Density": "The density of the bin the cell is in",
            "EstDensityForceX": "Estimated density gradient force (X) applied to cell",
            "EstDensityForceY": "Estimated density gradient force (Y) applied to cell",
            "EstDensityForceMag": "Magnitude of the estimated density force vector"
        }
        
        return res[["Iter", "CellId", "BinIdx", "Density", 
                    "EstDensityForceX", "EstDensityForceY", "EstDensityForceMag"]], desc

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
