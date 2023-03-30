import torch
from itertools import permutations
import itertools
from scipy.optimize import linear_sum_assignment

class LossFunc(torch.nn.Module):
    def __init__(self, device, num_query):
        super(LossFunc, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.device = device
        self.num_query = num_query

    def map_to_01(self, x):
        return -0.5 * (x - 1)

    def get_prob(self,cos_sim):
        return torch.softmax(cos_sim, -1)

    def get_all_permute(self):
        nums = list(range(0, self.num_query))
        return torch.tensor([list(p) for p in permutations(nums)]).to(self.device)

    @torch.no_grad()
    def match_cross_entropy(self, prob, label):
        permuta = self.get_all_permute()
        label_arr = []
        for permuta_cell in permuta:
            label_arr.append(permuta_cell.index_select(0, torch.flatten(label-1)).tolist())

        label_arr = torch.tensor(label_arr).to(self.device)
        loss_list = [self.cross_entropy(prob, x) for x in label_arr]
        min_index = loss_list.index(min(loss_list))
        target = label_arr[min_index]
        return target

    def k_means_loss(self, cos_sim):
        cos_sim = self.map_to_01(cos_sim[0])
        min_values, _ = torch.min(cos_sim, dim=1)
        cos_sim_loss = torch.sum(min_values) / len(min_values)
        return cos_sim_loss

    def center_loss(self, query_result):
        query_result = query_result[0]
        query_num, _ = query_result.shape
        similarity_matrix = torch.nn.functional.cosine_similarity(
                            query_result.unsqueeze(1), query_result.unsqueeze(0), dim=-1)
        traget = -1 * ((query_num) * (query_num-1) / 2)
        sum_cos = torch.sum(torch.triu(similarity_matrix, diagonal=1))
        loss = (sum_cos - traget) / (-1*traget)
        return loss

    def forward(self, cos_sim, label, query_result):
        '''
        :param cos_sim:[1, b_s*4, 4]
        :param labels:[b_s, 4]
        :return:
        '''
        prob = torch.softmax(cos_sim, -1)[0]
        # target_arr = self.match_cross_entropy(prob, label)
        # print('\n')
        # print(target_arr)
        # print(label)
        pre = torch.argmax(cos_sim.squeeze(0), dim=1).tolist()
        # print(pre)
        # print(label-1)
        cross_entropy_loss = self.cross_entropy(prob, label-1)
        center_point_loss = self.center_loss(query_result)
        return cross_entropy_loss + center_point_loss
