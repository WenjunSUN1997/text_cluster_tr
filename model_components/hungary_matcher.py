import torch
from scipy.optimize import linear_sum_assignment

class HungaryMatcher(object):

    def match(self,label_para, label_shape, label_classi, classification, paragraph_logits):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :return:
        '''
        label_para_flattened = label_para.flatten(0,1) #[b_s*max_len_arti, max_len_paragrapg]
        paragraph_prob = paragraph_logits.softmax(1)
        paragraph_prob_flattened = paragraph_prob.flatten(0,1) #[b_s*max_len_arti, max_len_paragrapg]
        paragraph_loss = torch.cdist(label_para_flattened, paragraph_prob_flattened, p=1)#[b_s*max_len_arti, max_len_paragrapg]

        label_classi_flattened = label_classi.flatten(0,1)#[b_s*max_len_arti, max_len_paragrapg]
        classification_flattened = classification.flatten(0,1)
        classi_loss = torch.cdist(label_classi_flattened, classification_flattened, p=1)

        all_loss = paragraph_loss + classi_loss
        batch_size, max_len_arti, _ = label_para.shape
        all_loss_batched = all_loss.view(batch_size, max_len_arti, -1)

        article_num = [v[0] for v in label_shape]
        indices = [linear_sum_assignment(c[i]) for i, c in
                   enumerate(all_loss_batched.split(article_num, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

    def match_per_batch(self, label_para, label_shape, label_classi,
                        classification, paragraph_logits):
        '''
        :param label_para: [b_s, max_len_arti, max_len_para], 1 for in the article
        :param label_classi: [b_s, max_len_arti, 2]
        :param label_shape: [b_s, 2] the shape the real label
        :param classification: [b_s, max_len_arti, 2]
        :param paragraph_logits: [b_s, max_len_arti, max_len_para]
        :return:
        '''
        final_loss = []
        paragraph_prob = paragraph_logits.softmax(1)
        batch_size = label_para.shape[0]
        for batch_index in range(batch_size):
            label_para_one_batch = label_para[batch_index]
            label_shape_one_batch = label_shape[batch_index]
            label_classi_one_batch = label_classi[batch_index]
            classification_one_batch = classification[batch_index]
            paragraph_prob_one_batch = paragraph_prob[batch_index]

            _, para_num = label_shape_one_batch
            label_para_real = label_para_one_batch[:, :para_num]
            paragraph_prob_real = paragraph_prob_one_batch[:, :para_num]
            paragraph_loss = torch.cdist(label_para_real, paragraph_prob_real, p=1)

            classi_loss = torch.cdist(label_classi_one_batch, classification_one_batch, p=1)

            all_loss = paragraph_loss + classi_loss
            row_ind, col_ind = linear_sum_assignment(all_loss)
            final_loss.append(all_loss[row_ind, col_ind].sum())

        return sum(final_loss)











