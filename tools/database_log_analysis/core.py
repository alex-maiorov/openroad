import sqlite3
import pandas as pd
import inspect
from typing import Tuple, Dict

class DatabaseLog:
    """Core interface to the OpenROAD database log using Pandas for data and sqlite3 for metadata."""
    
    def __init__(self, db_path: str):
        # Open in read-only mode, using URI=True for safe concurrent access
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._cache_schema()
        
    def _cache_schema(self):
        """Load available tables and metadata from the system tables using raw sqlite3."""
        self.tables = {}
        try:
            rows = self.conn.execute("SELECT table_name, column_names FROM table_list").fetchall()
            for r in rows:
                self.tables[r['table_name']] = r['column_names'].split(',')
        except sqlite3.OperationalError:
            pass # DB might be completely empty or missing system tables
            
        self.metadata = {}
        try:
            rows = self.conn.execute("SELECT key, value FROM metadata").fetchall()
            for r in rows:
                if r['key'] not in self.metadata:
                    self.metadata[r['key']] = []
                self.metadata[r['key']].append(r['value'])
        except sqlite3.OperationalError:
            pass

    def get_metadata(self, key: str = None):
        """Fetch metadata. If key is provided, return its values, else return all metadata."""
        if key:
            return self.metadata.get(key, [])
        return self.metadata

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a custom SQL query and return a Pandas DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Fetch an entire table by its user-provided name as a DataFrame."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found in database.")
        return self.query(f"SELECT * FROM [{table_name}]")


class AnalysisModule:
    """Base class for tool-specific analysis modules."""
    
    def __init__(self, db: DatabaseLog):
        self.db = db

    @classmethod
    def describe(cls):
        """Print descriptions of all available data and metrics in this module."""
        print(f"--- Module: {cls.__name__} ---")
        print(inspect.cleandoc(cls.__doc__ or "No module description."))
        print("\nAvailable Data & Metrics:")
        
        # Extract methods
        for name, func in inspect.getmembers(cls, predicate=inspect.isroutine):
            if name.startswith('_') or name == 'describe':
                continue
            doc = inspect.cleandoc(func.__doc__ or "No description provided.")
            print(f"  * {name}():\n      {doc.replace(chr(10), chr(10)+'      ')}")
        
        # Extract properties
        for name, prop in inspect.getmembers(cls, predicate=lambda o: isinstance(o, property)):
            if name.startswith('_'): continue
            doc = inspect.cleandoc(prop.__doc__ or "No description provided.")
            print(f"  * {name} (property):\n      {doc.replace(chr(10), chr(10)+'      ')}")
        print()
