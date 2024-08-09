import numpy as np

from .Infinite2DCanvas import Infinite2DCanvas


class RGBAInfiniteMixingCanvas:
    def __init__(self):
        self.NR = Infinite2DCanvas(0, 0, float)
        self.NG = Infinite2DCanvas(0, 0, float)
        self.NB = Infinite2DCanvas(0, 0, float)
        self.DSamples = Infinite2DCanvas(0, 0, int)

    def put(self, pixels: np.ndarray, x: int, y: int):
        H, W = pixels.shape[:2]

        R, G, B, A = [pixels[:, :, i].astype(float) / 255 for i in range(4)]
        scaled_R = np.multiply(R, A)
        scaled_G = np.multiply(G, A)
        scaled_B = np.multiply(B, A)

        existing_R = self.NR.get(x, y, W, H)
        existing_G = self.NG.get(x, y, W, H)
        existing_B = self.NB.get(x, y, W, H)

        combined_R = existing_R + scaled_R
        combined_G = existing_G + scaled_G
        combined_B = existing_B + scaled_B

        self.NR.put(combined_R, x, y)
        self.NG.put(combined_G, x, y)
        self.NB.put(combined_B, x, y)

        existing_DSamples = self.DSamples.get(x, y, W, H)
        new_DSamples = existing_DSamples + A.copy()
        self.DSamples.put(new_DSamples, x, y)

    def to_RGBA(self):
        DSamples_array = self.DSamples.to_array()
        deletion_mask = DSamples_array == 0
        modified_DSamples_array = DSamples_array.copy()
        modified_DSamples_array[deletion_mask] = 1
        R_array = self.NR.to_array() / modified_DSamples_array
        G_array = self.NG.to_array() / modified_DSamples_array
        B_array = self.NB.to_array() / modified_DSamples_array
        R_array[deletion_mask] = 0
        G_array[deletion_mask] = 0
        B_array[deletion_mask] = 0
        A_array = np.ones_like(R_array)
        A_array[deletion_mask] = 0
        return (
            np.dstack((R_array, G_array, B_array, A_array)).clip(0.0, 1.0) * 255
        ).astype(np.uint8)
