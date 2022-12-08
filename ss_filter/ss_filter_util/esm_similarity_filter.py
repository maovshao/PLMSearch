import torch
import torch.nn as nn
import torch.nn.functional as F

class esm_similarity_filiter():
    def __init__(self):
        pass

    def mse_mean_esm_identity_compute(self, esm_tensor1, esm_tensor2):        
        return (1 / F.mse_loss(esm_tensor1, esm_tensor2, reduction='sum'))

    def cos_mean_esm_identity_compute(self, esm_tensor1, esm_tensor2):        
        return F.cosine_similarity(esm_tensor1, esm_tensor2, dim=0)