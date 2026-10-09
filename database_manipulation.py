import numpy as np
import pandas as pd

def dataCleaning(df):
    df=pd.drop_duplicates()
    df=df.dropna()
    return df
    