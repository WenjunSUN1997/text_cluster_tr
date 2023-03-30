import os
from torch.utils.data.dataloader import DataLoader
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaModel
import pandas as pd
from model_config.tok_no_con import TokBertDiffer
from model_components.clustering_loss import LossFunc
from model_config.artr_encoder import ArtrEncoder
from transformers import BertModel
from model_config.artr_decoder import ArtrDecoder

class TextDataSeterCluster(torch.utils.data.Dataset):
    def __init__(self, csv_path, device, model_name):
        self.csv = pd.read_csv(csv_path)
        self.csv_seped = self.sep_csv()
        self.model_name = model_name
        if 'llama' not in model_name:
            self.bert = BertModel.from_pretrained(model_name).to(device)
        else:
            self.bert = LlamaModel.from_pretrained(model_name,
                                                   torch_dtype=torch.float16).to(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            self.tokenizer.add_special_tokens({'pad_token': 'no'})
        self.device = device

    def sep_csv(self):
        dfs = []
        csv_seped = self.csv.groupby('Class Index')
        for name, group in csv_seped:
            dfs.append(group.copy())
        min_len = min(len(x) for x in dfs)
        dfs = [x.head(min_len).reset_index(drop=True) for x in dfs]
        return dfs

    def __len__(self):
        return len(self.csv_seped[0])

    def get_bert_feature(self, item):
        text_cls = []
        text_tok = []
        for df_cell in self.csv_seped:
            try:
                text = df_cell['Title'][item] + ' ' + df_cell['Description'][item]
            except:
                try:
                    text = df_cell['Description'][item]
                except:
                    text = df_cell['Title'][item]

            output_toenizer = self.tokenizer([text], max_length=256,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='pt')
            if 'llama' in self.model_name:
                bert_feature = self.bert(input_ids=output_toenizer['input_ids']
                                         .to(self.device),
                                         attention_mask=output_toenizer['attention_mask']
                                         .to(self.device),
                                         output_hidden_states=True,
                                         return_dict=True
                                         )
                text_cls.append(bert_feature['last_hidden_state'].max(dim=1)[0].tolist())
                text_tok.append(bert_feature['last_hidden_state'].tolist())
            else:
                bert_feature = self.bert(input_ids=output_toenizer['input_ids']
                                         .to(self.device),
                                         attention_mask=output_toenizer['attention_mask']
                                         .to(self.device)
                                         )
                text_cls.append(bert_feature['pooler_output'].tolist())
                text_tok.append(bert_feature['last_hidden_state'].tolist())

        return torch.tensor(text_cls).squeeze(1).to(self.device), \
               torch.tensor(text_tok).squeeze(1).to(self.device)

    def get_label(self, item):
        label = []
        for df_cell in self.csv_seped:
            label.append(df_cell['Class Index'][item])

        return torch.tensor(label).to(self.device)

    def __getitem__(self, item):
        text_cls, text_tok = self.get_bert_feature(item)
        label = self.get_label(item)
        return {'text_cls': text_cls,
                'text_tok': text_tok,
                'label': label}

if __name__ == "__main__":
    csv = pd.read_csv('../data/train_yahoo.csv')
    x = TextDataSeterCluster(csv_path='../data/train_yahoo.csv', device='cuda:0')
    encoder = ArtrEncoder(hidd_dim=768, device='cuda:0')
    decoder = ArtrDecoder(num_obj_query=10, hidd_dim=768, device='cuda:0')
    encoder.to('cuda:0')
    decoder.to('cuda:0')
    batch_size = 2
    dataloader = DataLoader(x, batch_size=2)
    loss_func = LossFunc(device='cuda:0', num_query=10)
    for data in enumerate(dataloader):
        real = data[1]
        label = real['label']
        output = encoder(real['text_cls'].view(1,batch_size*10, 768),
                         real['text_tok'].view(1,batch_size*10, 256, 768 ))
        output_de = decoder(output)
        loss = loss_func(output_de['cos_sim'], label, output_de['query_result'])
        print(real)