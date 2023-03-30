import torch

class ArtrEncoder(torch.nn.Module):
    def __init__(self, hidd_dim, device):
        super(ArtrEncoder, self).__init__()
        self.hidd_dim = hidd_dim
        self.normalize = torch.nn.LayerNorm(normalized_shape=hidd_dim)
        self.device = device

        self.trans_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidd_dim,
                                                                    nhead=8,
                                                                    batch_first=True)
        self.trans_encoder = torch.nn.TransformerEncoder(self.trans_encoder_layer,
                                                         num_layers=2)
        self.linear = torch.nn.Linear(in_features=hidd_dim, out_features=hidd_dim)
        self.activation = torch.nn.Tanh()

    def encode(self, text_tok):
        result = self.trans_encoder_layer(src=text_tok)
        return result

    def forward(self, text_cls, text_tok):
        '''
        :param text_cls: [b_s, 4, 768]
        :param text_tok: [b_s, 4, 256, 768]
        :return: [b_s, 4, 768]
        '''
        semantic = torch.mean(text_tok, dim=2)
        semantic = self.encode(semantic)
        semantic = semantic + text_cls
        semantic = self.activation(self.linear(semantic))
        semantic = self.normalize(semantic)
        return semantic