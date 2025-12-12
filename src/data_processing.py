# src/data_processing.py  ← 100% WORKING VERSION – COPY EVERYTHING BELOW

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        agg = X.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            transaction_count=('Amount', 'count'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean'),
            std_value=('Value', 'std')
        ).reset_index()
        agg[['std_amount', 'std_value']] = agg[['std_amount', 'std_value']].fillna(0)
        return agg

class ExtractTimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['dt'] = pd.to_datetime(X['TransactionStartTime'], utc=True).dt.tz_convert('Africa/Kampala')
        time_feat = X.groupby('CustomerId').agg(
            avg_hour=('dt', lambda x: x.dt.hour.mean()),
            prop_weekend=('dt', lambda x: (x.dt.weekday >= 5).mean())
        ).reset_index()
        return time_feat

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        dfs = []
        for col in ['ProductCategory', 'ChannelId']:
            mode = X.groupby('CustomerId')[col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
            dfs.append(mode.rename(f'most_frequent_{col}').reset_index())
        top5 = X['ProductCategory'].value_counts().head(5).index
        for cat in top5:
            cnt = X[X['ProductCategory']==cat].groupby('CustomerId').size()
            dfs.append(cnt.rename(f'count_{cat.replace(" ","_")}').reset_index())
        result = dfs[0]
        for d in dfs[1:]:
            result = result.merge(d, on='CustomerId', how='left')
        return result.fillna(0)

def build_processing_pipeline():
    def pipeline(df):
        a = AggregateFeatures().transform(df)
        t = ExtractTimeFeatures().transform(df)
        c = CategoricalEncoder().transform(df)
        df = a.merge(t, on='CustomerId').merge(c, on='CustomerId')
        df = df.fillna(0)
        
        id_col = df['CustomerId']
        num_df = df.drop('CustomerId', axis=1)
        num_cols = num_df.select_dtypes(include=[np.number]).columns
        num_df[num_cols] = StandardScaler().fit_transform(num_df[num_cols])
        
        cat_cols = [col for col in num_df.columns if col.startswith('most_frequent_')]
        if cat_cols:
            enc = OneHotEncoder(sparse_output=False, drop='first')
            enc_data = enc.fit_transform(num_df[cat_cols])
            enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(), index=num_df.index)
            num_df = num_df.drop(cat_cols, axis=1)
            num_df = pd.concat([num_df, enc_df], axis=1)
        
        final = pd.concat([id_col, num_df], axis=1)
        return final
    return pipeline