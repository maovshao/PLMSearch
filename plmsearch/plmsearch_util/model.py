import torch
import torch.nn as nn
class plmsearch(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W z_j)
    """
    def __init__(self, embed_dim):
        super(plmsearch, self).__init__()
        self.projection_linear = nn.Linear(embed_dim, embed_dim)

    def load_pretrained(self, mtplm_path):
        #load to cpu at first, and then tranfer according to device_id
        state_dict = torch.load(mtplm_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def forward(self, z):
        h = self.projection_linear(z)
        return h