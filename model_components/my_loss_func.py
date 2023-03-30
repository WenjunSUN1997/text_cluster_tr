import torch

class LossFunc(torch.nn.Module):
    def __init__(self, ratio_cls:float, ratio_model_output:float):
        super(LossFunc, self).__init__()
        self.bert_no_pos = torch.nn.CosineEmbeddingLoss(margin=0.2)
        self.ar_with_pos = torch.nn.CosineEmbeddingLoss(margin=0.2)
        self.ratio_cls = ratio_cls
        self.ratio_model_output = ratio_model_output

    def forward(self, text_1_cls, text_2_cls,
                semantic_1_cls, semantic_2_cls, label):
        loss_bert_no_pos = self.bert_no_pos(text_1_cls, text_2_cls, label)
        loss_ar_with_pos = self.ar_with_pos(semantic_1_cls, semantic_2_cls, label)
        loss_all = self.ratio_cls * loss_bert_no_pos \
                   + self.ratio_model_output * loss_ar_with_pos
        return loss_all