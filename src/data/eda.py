from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

    def missing_values(self) -> None:
        df = pd.DataFrame(self.x.numpy())
        missing =  df.isnull().sum().sum()
        print("*"*20)
        print("Missing Values Check!")
        print("*"*20)
        print("Total Missing Values: {missing}")

        if missing == 0:
            print("Dataset is Clean")

    def label_distribution(self,save_path="figures/label_distribution.png") -> None:
        labels = self.y.numpy()
        plt.figure(figsize=(8,5))
        plt.hist(labels, bins=10, edgecolor="black", linewidth=1, histtype="stepfilled")
        plt.title("Distribution of Labels")
        plt.xlabel("Digit")
        plt.ylabel("Count")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path,dpi=300)
        plt.show()
        print(f"Saved -> {save_path}")
