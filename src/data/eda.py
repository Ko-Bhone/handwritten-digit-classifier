from pathlib import Path
import pandas as pd
import torch

class DataAnalyzer:
    def __init__(self,x:torch.Tensor,y:torch.Tensor):
        self.x = x
        self.y = y

    def dataset_info(self) -> None:
        df = pd.DataFrame(self.x.numpy())
        print("=" * 30)
        print("Dataset Information")
        print("=" * 30)
    
        print(f"Number of Samples : {df.shape[0]}")
        print(f"Number of Features: {df.shape[1]}")

        print("\nData Types")
        print(df.dtypes.value_counts())

        print("\nFirst Five Rows")
        print(df.head())
