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

class TextDataSeter(torch.utils.data.Dataset):
    def __init__(self, csv_path, device):
        self.csv = pd.read_csv(csv_path)
        self.csv_seped = self.sep_csv()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device

    def sep_csv(self):
        news_1 = self.csv[self.csv['Class Index'] == 1].reset_index(drop=True)
        news_2 = self.csv[self.csv['Class Index'] == 2].reset_index(drop=True)
        news_3 = self.csv[self.csv['Class Index'] == 3].reset_index(drop=True)
        news_4 = self.csv[self.csv['Class Index'] == 4].reset_index(drop=True)
        return news_1, news_2, news_3, news_4

    def __len__(self):
        return len(self.csv_seped[0])

    def get_bert_feature(self, text_item):
        text_cls = []
        text_tok = []
        for index in range(len(text_item)):
            output_toenizer = self.tokenizer([text_item[index]], max_length=256,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='pt')
            bert_feature = self.bert(input_ids=output_toenizer['input_ids']
                                     .to(self.device),
                                     attention_mask=output_toenizer['attention_mask']
                                     .to(self.device))
            text_cls.append(bert_feature['pooler_output'].tolist())
            text_tok.append(bert_feature['last_hidden_state'].tolist())

        return torch.tensor(text_cls).squeeze(1).to(self.device), \
               torch.tensor(text_tok).squeeze(1).to(self.device)

    def get_label(self, label):
        label_final = torch.zeros((4,len(label))).to(self.device)
        for index in range(len(label)):
            label_final[index][label[index]-1] = 1

        return label_final

    def __getitem__(self, item):
        news_1, news_2, news_3, news_4 = self.csv_seped
        news_1_item = news_1['Title'][item] + ' ' +news_1['Description'][item]
        news_2_item = news_2['Title'][item] + ' ' +news_2['Description'][item]
        news_3_item = news_3['Title'][item] + ' ' +news_3['Description'][item]
        news_4_item = news_4['Title'][item] + ' ' +news_4['Description'][item]
        text_item = [news_1_item, news_2_item, news_3_item, news_4_item]
        text_cls, text_tok = self.get_bert_feature(text_item)
        label = [news_1['Class Index'][item],
                 news_2['Class Index'][item],
                 news_3['Class Index'][item],
                 news_4['Class Index'][item]]
        label = torch.tensor(label).to(self.device)
        # label = self.get_label(label)
        return {'text_cls': text_cls,
                'text_tok': text_tok,
                'label': label}

if __name__ == "__main__":
    csv = pd.read_csv('../data/test.csv')
    x = TextDataSeter(csv_path='../data/test.csv', device='cuda:0')
    encoder = ArtrEncoder(hidd_dim=768, device='cuda:0')
    decoder = ArtrDecoder(num_obj_query=4, hidd_dim=768, device='cuda:0')
    encoder.to('cuda:0')
    decoder.to('cuda:0')
    batch_size = 2
    dataloader = DataLoader(x, batch_size=2)
    loss_func = LossFunc()
    for data in enumerate(dataloader):
        real = data[1]
        label = real['label']
        output = encoder(real['text_cls'].view(1,batch_size*4, 768),
                         real['text_tok'].view(1,batch_size*4, 256, 768 ))
        output_de = decoder(output)
        loss = loss_func(output_de['cos_sim'], label, output_de['query_result'])
        print(real)