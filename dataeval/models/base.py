# dataeval/models/base.py

class ModelAdapter:
    def extract_batch(self, batch_frames):
        """
        batch_frames: List[np.ndarray], each shape [T, H, W, 3]
        return: np.ndarray [B, D]
        """
        raise NotImplementedError

    def extract(self, frames):
        """
        单样本兼容接口，默认走 batch
        """
        return self.extract_batch([frames])[0]
