import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

class DataPreprocessor:

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть типа pandas.DataFrame")
        
        self.df = df.copy()
        self.history: Dict[str, Any] = {
            'removed_columns': [],
            'imputation_values': {},
            'categorical_columns': [],
            'new_one_hot_columns': [],
            'normalization_params': {}
        }

    def remove_missing(self, threshold: float = 0.5, strategy: str = 'mean'):
        missing_rate = self.df.isnull().mean()
        cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
        self.history['removed_columns'] = cols_to_drop
        self.df.drop(columns=cols_to_drop, inplace=True)

        for col in self.df.columns:
            if self.df[col].isnull().any():
                is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
                
                if not is_numeric:
                    val = self.df[col].mode()[0]
                else:
                    if strategy == 'mean':
                        val = self.df[col].mean()
                    elif strategy == 'median':
                        val = self.df[col].median()
                    else:
                        val = self.df[col].mode()[0]
                
                self.history['imputation_values'][col] = val
                self.df[col] = self.df[col].fillna(val)
        
        return self.df

    def encode_categorical(self) -> pd.DataFrame:
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.history['categorical_columns'] = cat_cols

        if cat_cols:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=False, dtype=int)
        
        self.history['new_one_hot_columns'] = self.df.columns.tolist()
        return self.df

    def normalize_numeric(self, method: str = 'minmax') -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            
            if method == 'minmax':
                c_min, c_max = self.df[col].min(), self.df[col].max()
                self.history['normalization_params'][col] = {'min': c_min, 'max': c_max}
                if c_max - c_min != 0:
                    self.df[col] = (self.df[col] - c_min) / (c_max - c_min)
            
            elif method == 'std':
                mean, std = self.df[col].mean(), self.df[col].std()
                self.history['normalization_params'][col] = {'mean': mean, 'std': std}
                if std != 0:
                    self.df[col] = (self.df[col] - mean) / std
        
        return self.df

    def fit_transform(self, threshold: float = 0.5, strategy: str = 'mean', method: str = 'minmax') -> pd.DataFrame:
        self.remove_missing(threshold=threshold, strategy=strategy)
        self.encode_categorical()
        self.normalize_numeric(method=method)
        return self.df