import torch
from model_config.artr_encoder import ArtrEncoder
from model_config.artr_decoder import ArtrDecoder

class Artr(torch.nn.Module):
    def __init__(self, num_obj_query, hidd_dim, device):
        super(Artr, self).__init__()
        self.encoder = ArtrEncoder(hidd_dim, device)
        self.decoder = ArtrDecoder(num_obj_query, hidd_dim, device)
        self.device = device

    def forward(self, text_cls, text_tok):
        memory = self.encoder(text_cls, text_tok)
        result = self.decoder(memory)
        return result



