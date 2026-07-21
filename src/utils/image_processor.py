from PIL import Image
import numpy as np
import torch

class ImageProcessor:

    @torch.no_grad()
    def preprocess(image: Image.Image) -> torch.Tensor:
        image = image.convert('L')
        image = image.resize((20,20))
        image = np.array(image,dtype=np.float32)
        image /= 255.0
        image = image.reshape(1,400)
        return torch.tensor(image)
