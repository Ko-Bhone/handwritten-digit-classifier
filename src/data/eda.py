from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import numpy as np

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
        unique , counts = np.unique(labels, return_counts=True)
        bars = plt.bar(unique,counts,edgecolor="black",linewidth=1)
        plt.title("Label Distribution",fontsize=18)
        plt.xlabel("Digit",fontsize=14)
        plt.ylabel("Number of images",fontsize=14)
        plt.xticks(unique)
        for bar,count in zip(bars,counts):
            plt.text(bar.get_x() + bar.get_width()/2,count+5,str(count),ha="center",fontsize=10)
        plt.grid(axis='y',linestyle="--",alpha=0.3)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path,dpi=300)
        plt.show()

    def show_random_samples(self,num_samples:int=25,save_path="figures/random_samples.png") -> None:
        fig,axes  = plt.subplots(5,5,figsize=(8,8))
        indices = random.sample(range(len(self.x)),num_samples)

        for ax, index in zip(axes.flat, indices):
            image = self.x[index].reshape(20,20).T.numpy()
            ax.imshow(image,cmap="gray")
            ax.set_title(f"Label:{self.y[index].item()}",fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path,dpi=300)
        plt.show()
        print(f"Saved -> {save_path}")

    def pixel_statics(self) -> None:
        print("\n" + "=" * 50)
        print("Pixel Statistics")
        print("=" * 50)

        print(f"Minimum Pixel Value : {self.x.min().item():.2f}")
        print(f"Maximum Pixel Value : {self.x.max().item():.2f}")
        print(f"Mean Pixel Value    : {self.x.mean().item():.2f}")
        print(f"Std Pixel Value     : {self.x.std().item():.2f}")

    def average_digit_image(self,save_path:str="figures/average_digit_image.png") -> None:
        average_image = self.x.mean(dim=0).reshape(20,20).T.numpy()
        plt.figure(figsize=(5,5))
        plt.imshow(average_image,cmap="gray")
        plt.title("Average Digit ")
        plt.axis("off")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path,dpi=300)
        plt.show()
        print(f"Saved -> {save_path}")