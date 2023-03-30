import os
from torch.utils.data.dataloader import DataLoader
import torch
from transformers import AutoTokenizer
import pandas as pd
from model_config.tok_no_con import TokBertDiffer
from model_components.clustering_loss import LossFunc
from model_config.artr_encoder import ArtrEncoder
from transformers import BertModel
from model_config.artr_decoder import ArtrDecoder

class TextDataSeterYahoo(torch.utils.data.Dataset):
    def __init__(self, csv_path, device):
        self.csv = pd.read_csv(csv_path)
        self.csv.columns = ['Class Index', 'Title', 'temp', 'Description']
        self.csv_seped = self.sep_csv()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

    def sep_csv(self):
        dfs = []
        csv_seped = self.csv.groupby('Class Index')
        for name, group in csv_seped:
            dfs.append(group.copy())
        min_len = min(len(x) for x in dfs)
        dfs = [x.head(min_len).reset_index() for x in dfs]
        return dfs

    def __len__(self):
        return len(self.csv_seped[0])

    def get_bert_feature(self, text_item):

        output_toenizer = self.tokenizer([text_item], max_length=256,
                                         truncation=True,
                                         padding='max_length',
                                         return_tensors='pt')
        bert_feature = self.bert(input_ids=output_toenizer['input_ids']
                                 .to(self.device),
                                 attention_mask=output_toenizer['attention_mask']
                                 .to(self.device))
        text_cls = bert_feature['pooler_output'].tolist()
        text_tok = bert_feature['last_hidden_state'].tolist()

        return torch.tensor(text_cls).squeeze(1).to(self.device), \
               torch.tensor(text_tok).squeeze(1).to(self.device)

    def get_label(self, label):
        label_final = torch.zeros((4,len(label))).to(self.device)
        for index in range(len(label)):
            label_final[index][label[index]-1] = 1

        return label_final

    def __getitem__(self, item):
        text_item = str(self.csv['Title'][item]) + ' ' \
                    +str(self.csv['Description'][item])
        text_cls, text_tok = self.get_bert_feature(text_item)
        label = int(self.csv['Class Index'][item])
        label = torch.tensor(label).to(self.device)
        return {'text_cls': text_cls,
                'text_tok': text_tok,
                'label': label}

if __name__ == "__main__":
    csv = pd.read_csv('../data/yahootrain.csv')
    csv.columns = ['Class Index', 'Title', 'temp', 'Description']
    x = TextDataSeterYahoo(csv_path='../data/yahootest.csv', device='cuda:0')
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