import torch

class TokBertConDiffer(torch.nn.Module):
    def __init__(self, bert_model, hidd_dim):
        super(TokBertConDiffer, self).__init__()
        self.bert_model = bert_model
        self.trans_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidd_dim,
                                                                    nhead=8)
        self.trans_encoder = torch.nn.TransformerEncoder(self.trans_encoder_layer,
                                                         num_layers=2)
        self.trans_encoder_layer_all = torch.nn.TransformerEncoderLayer(d_model=hidd_dim,
                                                                    nhead=8)
        self.trans_encoder_all = torch.nn.TransformerEncoder(self.trans_encoder_layer,
                                                         num_layers=2)
        self.x_pos_embedding = torch.nn.Embedding(3000, hidd_dim)
        self.y_pos_embedding = torch.nn.Embedding(5000, hidd_dim)
        self.linear_cell = torch.nn.Linear(in_features=hidd_dim, out_features=hidd_dim)
        self.activation_cell = torch.nn.Tanh()
        self.linear_all = torch.nn.Linear(in_features=hidd_dim, out_features=hidd_dim)
        self.activation_all = torch.nn.Tanh()

    def get_bert_future(self, text_1, text_2, text_all):
        text_1_bert = self.bert_model(input_ids=text_1['input_ids'].squeeze(1),
                                      attention_mask=text_1['attention_mask'].squeeze(1))
        text_2_bert = self.bert_model(input_ids=text_2['input_ids'].squeeze(1),
                                      attention_mask=text_2['attention_mask'].squeeze(1))
        text_all_bert = self.bert_model(input_ids=text_all['input_ids'].squeeze(1),
                                      attention_mask=text_all['attention_mask'].squeeze(1))
        text_1_cls = text_1_bert['pooler_output']
        text_2_cls = text_2_bert['pooler_output']
        text_all_cls = text_all_bert['pooler_output']
        text_1_tok = text_1_bert['last_hidden_state']
        text_2_tok = text_2_bert['last_hidden_state']
        text_all_tok = text_all_bert['last_hidden_state']

        return text_1_cls, text_2_cls, text_all_cls, \
               text_1_tok, text_2_tok, text_all_tok

    def forward_position(self, x, y):
        x_pos_embedding = self.x_pos_embedding(x)
        y_pos_embedding = self.y_pos_embedding(y)
        return x_pos_embedding, y_pos_embedding

    def forward_text_all(self, text_1_tok, text_2_tok, text_all_tok,
                         x_1, y_1, x_2, y_2):
        x_1_embedding, y_1_embedding = self.forward_position(x_1, y_1)
        x_2_embedding, y_2_embedding = self.forward_position(x_2, y_2)
        x_all_embedding = (x_1_embedding + x_2_embedding) / 2
        y_all_embedding = (y_1_embedding + y_2_embedding) / 2

        semantic_1_middle = text_1_tok + x_1_embedding.unsqueeze(1) \
                            + y_1_embedding.unsqueeze(1)
        semantic_2_middle = text_2_tok + x_2_embedding.unsqueeze(1) \
                            + y_2_embedding.unsqueeze(1)
        semantic_all_middle = text_all_tok + x_all_embedding.unsqueeze(1) \
                              + y_all_embedding.unsqueeze(1)

        semantic_1 = self.trans_encoder(semantic_1_middle)
        semantic_2 = self.trans_encoder(semantic_2_middle)
        semantic_all = self.trans_encoder_all(semantic_all_middle)

        semantic_1_cls = self.activation_cell(self.linear_cell(semantic_1[:, 0, :]))
        semantic_2_cls = self.activation_cell(self.linear_cell(semantic_2[:, 0, :]))
        semantic_all_cls = self.activation_all(self.linear_all(semantic_all[:, 0, :]))

        sematic_1_final = semantic_1_cls + (semantic_all_cls - semantic_2_cls)
        sematic_2_final = semantic_2_cls + (semantic_all_cls - semantic_1_cls)
        return sematic_1_final, sematic_2_final

    def forward(self, text_1, text_2, text_all, x_1, y_1, x_2, y_2):
        text_1_cls, text_2_cls, text_all_cls, text_1_tok, text_2_tok, text_all_tok = \
            self.get_bert_future(text_1, text_2, text_all)
        sematic_1_final, sematic_2_final = self.forward_text_all(
            text_1_tok, text_2_tok, text_all_tok, x_1, y_1, x_2, y_2)

        return text_1_cls, text_2_cls, sematic_1_final, sematic_2_final


