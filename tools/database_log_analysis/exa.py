import pandas as pd
from typing import Tuple, Dict, Any, List
from .core import AnalysisModule

class ExaAnalysis(AnalysisModule):
    """
    Analysis module for the EXA (Example) tool.
    Provides access to raw logged tables as Pandas DataFrames and demonstrates
    how to derive metrics from them.
    
    All properties and methods that return dataframes will return a tuple of:
    (pd.DataFrame, dict_of_column_descriptions)
    """

    @property
    def st_nonbulk(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Single-threaded, non-bulk logged values.
        """
        df = self.db.get_table("exa_st_nonbulk")
        desc = {
            "id": "Sequential identifier for the logged value",
            "val": "Random float value between 0.0 and 1.0"
        }
        return df, desc

    @property
    def st_bulk(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Single-threaded, bulk logged values.
        """
        df = self.db.get_table("exa_st_bulk")
        desc = {
            "id": "Sequential identifier for the logged value",
            "val": "Random float value between 0.0 and 1.0",
            "type": "Random integer classification type (0-100)"
        }
        return df, desc

    @property
    def mt_nonbulk(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Multi-threaded, non-bulk logged values.
        """
        df = self.db.get_table("exa_mt_nonbulk")
        desc = {
            "thread_id": "ID of the thread that generated the log",
            "iter": "Iteration index within the thread",
            "val": "Random float value between 0.0 and 1.0"
        }
        return df, desc

    @property
    def mt_bulk(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Raw Data: Multi-threaded, bulk logged values.
        """
        df = self.db.get_table("exa_mt_bulk")
        desc = {
            "thread_id": "ID of the thread that generated the log",
            "iter": "Iteration index within the thread",
            "val": "Random float value between 0.0 and 1.0"
        }
        return df, desc

    def get_average_st_bulk_val(self) -> float:
        """
        Derived Data: Calculates the average 'val' across all single-threaded bulk logs.
        """
        df, _ = self.st_bulk
        if df.empty:
            return 0.0
        return float(df['val'].mean())

    def calculate_magic_metric(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Derived Data: A contrived metric that groups single-threaded bulk logs
        by 'type' and calculates a 'magic score' (mean of val * 100).
        """
        df, _ = self.st_bulk
        if df.empty:
            return pd.DataFrame(), {}
        
        # Calculate magic metric using pandas
        magic_df = df.groupby('type', as_index=False)['val'].mean()
        magic_df['magic_score'] = magic_df['val'] * 100
        
        # Keep only the relevant columns
        magic_df = magic_df[['type', 'magic_score']]
        
        desc = {
            "type": "Classification type from the single-threaded bulk data",
            "magic_score": "Contrived metric representing the mean value scaled by 100"
        }
        return magic_df, desc

    @property
    def test_metadata(self) -> List[str]:
        """
        Metadata: Returns the value(s) associated with 'test_metadata_key' logged by EXA.
        """
        return self.db.metadata.get("test_metadata_key", [])
