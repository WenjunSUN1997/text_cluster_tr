import torch
from sklearn import metrics
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, precision_score, recall_score, f1_score

@torch.no_grad()
def validate(model, dataloader, loss_func, semantic_dim, max_len_para):
    acc = []
    nmi = []
    ari = []
    pre_all = []
    label_all = []
    loss_all = []
    for data in tqdm(dataloader):
        label = torch.flatten(data['label'])
        text_cls = data['text_cls']
        text_tok = data['text_tok']
        b_s, num_sen, _ = text_cls.shape
        text_cls = text_cls.view(1, b_s * num_sen, semantic_dim)
        text_tok = text_tok.view(1, b_s * num_sen, max_len_para, semantic_dim)
        output = model(text_cls, text_tok)
        query_result = output['query_result']
        cos_sim = output['cos_sim']
        loss = loss_func(cos_sim, label, query_result)
        loss_all.append(loss.item())
        label_for_eva = (torch.flatten(label)-1).tolist()
        # print(label_for_eva)
        label_all += label_for_eva
        pre = torch.argmax(cos_sim.squeeze(0), dim=1).tolist()
        # print(pre)
        pre_all += pre
        acc_cell = acc_func(label_for_eva, pre)
        acc.append(acc_cell)
        ari_cell = adjusted_rand_score(labels_true=label_for_eva, labels_pred=pre)
        ari.append(ari_cell)
        nmi_cell = normalized_mutual_info_score(labels_true=label_for_eva, labels_pred=pre)
        nmi.append(nmi_cell)
        # break
    print('loss', sum(loss_all) / len(loss_all))
    print('acc', sum(acc) / len(acc))
    print('ari', sum(ari) / len(ari))
    print('nmi', sum(nmi) / len(nmi))
    print('acc', acc_func(label_all, pre_all))
    print('ari', adjusted_rand_score(labels_true=label_all, labels_pred=pre_all))
    print('nmi', normalized_mutual_info_score(labels_true=label_all, labels_pred=pre_all))
    print('acc', accuracy_score(y_true=label_all, y_pred=pre_all))
    print('p', precision_score(y_true=label_all, y_pred=pre_all, average='micro'))
    print('r', recall_score(y_true=label_all, y_pred=pre_all, average='micro'))
    print('f', f1_score(y_true=label_all, y_pred=pre_all, average='micro'))

    return sum(loss_all)/len(loss_all), sum(acc)/len(acc)

def acc_func(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size







