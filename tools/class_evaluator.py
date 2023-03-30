import numpy as np
from collections import defaultdict
from collections import Counter

class Evaluator(object):
    def __init__(self, pre:list, label:list):
        self.pre = pre
        self.lable = label
        self.p_matrix = self.build_p_matric()
        self.r_matrix = self.build_r_matric()

    def group_item(self, cluster_result):
        d = defaultdict(list)
        for i, x in enumerate(cluster_result):
            d[x].append(i)

        return list(d.values())

    def build_r_matric(self):
        gt = self.group_item(self.lable)
        hy = self.group_item(self.pre)
        matrix = np.zeros((len(hy), len(gt)))
        for index_x in range(len(hy)):
            for index_y in range(len(gt)):
                counter1 = Counter(gt[index_y])
                counter2 = Counter(hy[index_x])
                intersection = counter1 & counter2
                same_num = sum(intersection.values())
                matrix[index_x][index_y] = same_num / len(gt[index_y])

        return matrix

    def build_p_matric(self):
        gt = self.group_item(self.lable)
        hy = self.group_item(self.pre)
        matrix = np.zeros((len(hy), len(gt)))
        for index_x in range(len(hy)):
            for index_y in range(len(gt)):
                counter1 = Counter(gt[index_y])
                counter2 = Counter(hy[index_x])
                intersection = counter1 & counter2
                same_num = sum(intersection.values())
                matrix[index_x][index_y] = same_num / len(hy[index_x])

        return matrix

    def greedy(self, B):
        sum = 0
        B_prime = B.copy()
        while B_prime.size > 0:
            b = np.max(B_prime)
            sum += b
            idx = np.unravel_index(np.argmax(B_prime), B_prime.shape)
            i, j = idx
            B_prime = np.delete(np.delete(B_prime, i, axis=0), j, axis=1)
        return sum

    def get_r_value(self):
        num = self.r_matrix.shape[0]
        sum = self.greedy(self.r_matrix)
        return sum/num

    def get_p_value(self):
        num = self.p_matrix.shape[1]
        sum = self.greedy(self.p_matrix)
        return sum/num

# if __name__ == "__main__":
#     obj = Evaluator([1,1,1,2,2,2,3,3,3],[1,1,1,2,2,2])
#     c = obj.build_p_matric()
#     a = obj.get_r_value()
#     print(a)