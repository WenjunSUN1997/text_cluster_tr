import torch
from transformers import AutoTokenizer
from ast import literal_eval

class TextDataSeter(torch.utils.data.Dataset):
    def __init__(self, data_csv, lang, device):
        self.data_csv = data_csv
        self.device = device
        if lang == 'fre':
            self.tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        if lang == 'fin':
            self.tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/bert-base-finnish-cased-v1')

    def __len__(self):
        return len(self.data_csv['label'])

    def normalize_bbox(self, bbox):
        return [
            int(3000 * (bbox[0] / 10000)),
            int(5000 * (bbox[1] / 10000)),
            int(3000 * (bbox[2] / 10000)),
            int(5000 * (bbox[3] / 10000)),
        ]

    def get_output_tokenizer(self, text):
        output_tokenizer = self.tokenizer(text, max_length=512,
                                           truncation=True,
                                           padding='max_length',
                                          return_tensors = 'pt')
        output_tokenizer_gpu = {'input_ids':output_tokenizer['input_ids'].to(self.device),
                                'attention_mask':output_tokenizer['attention_mask'].to(self.device)}

        return output_tokenizer_gpu

    def __getitem__(self, item):
        text_1 = self.data_csv['text_1'][item]
        text_2 = self.data_csv['text_2'][item]
        text_all = self.data_csv['text_all'][item]

        bbox_1 = literal_eval(self.data_csv['bbox_1'][item])
        bbox_1 = self.normalize_bbox(bbox_1)
        x_1 = int((bbox_1[0] + bbox_1[2]) / 2)
        if x_1 >= 10000:
            x_1 = 10000
        y_1 = int((bbox_1[1] + bbox_1[3]) / 2)
        if y_1 >= 10000:
            y_1 = 10000
        bbox_2 = literal_eval(self.data_csv['bbox_2'][item])
        bbox_2 = self.normalize_bbox(bbox_2)
        x_2 = int((bbox_2[0] + bbox_2[2]) / 2)
        if x_2 >= 10000:
            x_2 = 10000
        y_2 = int((bbox_2[1] + bbox_2[3]) / 2)
        if y_2 >= 10000:
            y_2 = 10000

        label = int(self.data_csv['label'][item])
        if label == 0 :
            label = -1

        text_1_tokenizer = self.get_output_tokenizer(text_1)
        text_2_tokenizer = self.get_output_tokenizer(text_2)
        text_all_tokenizer = self.get_output_tokenizer(text_all)

        return {'text_1':text_1_tokenizer,
                'text_2':text_2_tokenizer,
                'text_all':text_all_tokenizer,
                'x_1' : x_1,
                 'y_1' :y_1,
                'x_2' : x_2,
                'y_2' : y_2,
                'label':label}