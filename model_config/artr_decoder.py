import torch

class ArtrDecoder(torch.nn.Module):
    def __init__(self, num_obj_query, hidd_dim, device):
        super(ArtrDecoder, self).__init__()
        self.device = device
        self.num_obj_query = num_obj_query
        self.obj_query_embedding = torch.nn.Embedding(num_obj_query, hidd_dim)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=hidd_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=2)

    def get_obj_query_embedding(self, batch_size):
        obj_query_index = torch.tensor([x for x in range(self.num_obj_query)])
        obj_query_embedding = self.obj_query_embedding(obj_query_index.to(self.device))
        obj_query_embedding_batched = obj_query_embedding.repeat(batch_size, 1, 1)
        return obj_query_embedding_batched

    def forward_obj_query(self, text_embedding):
        '''
        :param text_embedding: padded text embedding of one newspaper [b_s, 500, hidd_dim]
        :param mask: attention mask [b_s, 500] 0 for no_masked, 1 for masked
        :param obj_query_embedding: [b_s, num_obj_query, hidd_dim]
        :return: query_result: [b_s, num_obj_query, hidd_dim]
        '''
        obj_query_embedding_batched = self.get_obj_query_embedding(text_embedding.shape[0])
        query_result = self.decoder(memory=text_embedding,
                                     tgt=obj_query_embedding_batched)
        return query_result

    def get_cos_sim(self, query_result, text_embedding):
        cos_sim = torch.cosine_similarity(query_result.unsqueeze(1),
                                          text_embedding.unsqueeze(-2),
                                          dim=-1)
        return cos_sim

    def forward(self, text_embedding):
        query_result = self.forward_obj_query(text_embedding)
        cos_sim = self.get_cos_sim(query_result, text_embedding)
        return {'query_result': query_result,
                'cos_sim': cos_sim}



