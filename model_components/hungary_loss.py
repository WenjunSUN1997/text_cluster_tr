import torch
from scipy.optimize import linear_sum_assignment

class HungaryLoss(torch.nn.Module):
    def __init__(self, no_article_weight=0.1, device='cuda:0'):
        super(HungaryLoss, self).__init__()
        self.no_article_weight = no_article_weight
        self.classi_loss = []
        self.para_loss = []
        self.device = device
        self.no_article_weight = no_article_weight

    def create_mask_para(self, label_para, label_shape):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para]
        :param label_shape: [b_s, 2] the shape the real label
        :return: mask_list [b_s, [max_len_arti,max_len_para]], 1 for remaining
        '''
        b_s, max_len_arti, max_len_para = label_para.shape
        mask = []
        for batch_index in range(b_s):
            len_para = label_shape[batch_index][1]
            mask_temp = torch.zeros(max_len_arti, max_len_para)
            mask_temp[:, :len_para] = 1
            mask.append(mask_temp.to(self.device))

        return mask

    def create_mask_class(self, label_para, label_shape):
        b_s, max_len_arti, max_len_para = label_para.shape
        mask = []
        for batch_index in range(b_s):
            len_ar = label_shape[batch_index][0]
            mask_temp = torch.ones(max_len_arti, 2)
            mask_temp[len_ar:, :] = self.no_article_weight
            mask.append(mask_temp.to(self.device))

        return mask

    @torch.no_grad()
    def match(self, label_para, label_shape, label_classi,
                        classification, paragraph_logits, cos_sim):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :param cos_sim: [b_s, max_len_arti, max_len_para]
        :return:
        '''
        final_loss = []
        paragraph_prob = paragraph_logits.softmax(1)
        batch_size = label_para.shape[0]
        row_list = []
        column_list = []
        for batch_index in range(batch_size):
            article_num, para_num = label_shape[batch_index]
            cos_sim_real = cos_sim[batch_index][:, :para_num]
            label_para_real = label_para[batch_index][:, :para_num]
            paragraph_prob_real = paragraph_prob[batch_index][:, :para_num]
            paragraph_loss = torch.cdist(paragraph_prob_real,
                                              label_para_real,
                                              p=1)
            classi_loss = torch.cdist(classification[batch_index],
                                           label_classi[batch_index],
                                           p=1)
            label_cos_sim_real = torch.where(label_para_real == 0,
                                             torch.tensor(-1).to(self.device),
                                             label_para_real)
            cos_sim_loss = torch.cdist(cos_sim_real,
                                        label_cos_sim_real,
                                        p=1)

            all_loss_cell = (cos_sim_loss + classi_loss).detach().cpu()
            row_ind, col_ind = linear_sum_assignment(all_loss_cell)
            row_list.append(row_ind)
            column_list.append(col_ind)
            final_loss.append(all_loss_cell[row_ind, col_ind].sum())

        return row_list, column_list

    def forward(self, label_para, label_shape, label_classi,
                        classification, paragraph_logits, cos_sim):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :return: loss
        '''
        b_s, max_len_arti, max_len_para = label_para.shape
        mask_para = self.create_mask_para(label_para, label_shape)
        mask_para = torch.cat(mask_para, dim=0)
        mask_class = self.create_mask_class(label_para, label_shape)
        # mask_class = torch.cat(mask_class, dim=0)
        row_list, column_list = self.match(label_para, label_shape, label_classi,
                        classification, paragraph_logits, cos_sim)
        label_para = torch.stack([label_para[x][column_list[x]]
                                  for x in range(b_s)], dim=0).view(b_s * max_len_arti, -1)
        label_classi = torch.stack([label_classi[x][column_list[x]]
                                  for x in range(b_s)], dim=0).view(b_s * max_len_arti, -1)
        mask_class = torch.stack([mask_class[x][column_list[x]]
                                  for x in range(b_s)], dim=0).view(b_s * max_len_arti, -1)
        label_cos_sim = torch.where(label_para == 0,
                                    torch.tensor(-1).to(self.device),
                                    label_para)

        paragraph_prob = paragraph_logits.softmax(1).view(b_s * max_len_arti, -1)
        classification = classification.view(b_s * max_len_arti, -1)
        row_pre_as_ar = torch.nonzero(classification[:, 1] > classification[:, 0],
                                      as_tuple=False)[:, 0]
        cos_sim = cos_sim.view(b_s * max_len_arti, -1)
        para_loss = torch.sum(torch.abs(paragraph_prob-label_para)*mask_para)
        cos_loss = torch.sum((torch.abs(cos_sim-label_cos_sim)*mask_para)[row_pre_as_ar])
        class_loss = torch.sum(torch.abs(classification-label_classi)*mask_class)

        return class_loss, cos_loss
