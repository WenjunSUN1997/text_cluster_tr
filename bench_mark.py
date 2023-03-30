import torch
from model_components.validator_artr import validate
from model_config.artr import Artr
import argparse
from model_components import dataloader_artr
from sklearn.cluster import KMeans
from tqdm import tqdm
from model_components.validator_artr import acc_func
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

def train_model(dataset, max_len_arti, semantic_dim, model_name,
                device, save_path, max_len_para, store_threshold):
    LR = 5e-5
    epoch = 10000
    loss_all = []
    acc = []
    ari = []
    nmi = []
    kmeans = KMeans(n_clusters=max_len_arti)
    test_dataloader = dataloader_artr.get_dataloader(goal='test',
                                                      device=device,
                                                      dataset_name=dataset,
                                                      batch_size=batch_size,
                                                      model_name=model_name)
    for data in tqdm(test_dataloader):
        label = torch.flatten(data['label']).tolist()
        # print(label)
        text_cls = data['text_cls']
        b_s, num_sen, _ = text_cls.shape
        text_cls = text_cls.view(1, b_s*num_sen, semantic_dim)[0].tolist()
        kmeans.fit(text_cls)
        pre_labels = kmeans.labels_
        acc.append(acc_func(y_true=label, y_pred=pre_labels))
        ari.append(adjusted_rand_score(labels_true=label, labels_pred=pre_labels))
        nmi.append(normalized_mutual_info_score(labels_true=label, labels_pred=pre_labels))

    print('acc', sum(acc)/len(acc))
    print('ari', sum(ari) / len(ari))
    print('nmi', sum(nmi) / len(nmi))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='agnews')
    parser.add_argument("--store_threshold", default=0.75)
    parser.add_argument("--save_path", default='bbc')
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--model_type", default='no_con')
    parser.add_argument("--max_len_para", default=256)
    parser.add_argument("--max_len_arti", default=5)
    parser.add_argument("--cluster_num", default=4)
    parser.add_argument("--model_name", default='bert-base-uncased')
    parser.add_argument("--semantic_dim", default=768)
    parser.add_argument("--model_path", default='model_zoo/TokBertDiffer_fin_4_192000.pth')
    args = parser.parse_args()
    print(args)
    store_threshold = float(args.store_threshold)
    save_path = args.save_path
    dataset = args.dataset
    cluster_num = int(args.cluster_num)
    batch_size = int(args.batch_size)
    device = args.device
    model_type = args.model_type
    semantic_dim = int(args.semantic_dim)
    model_path = args.model_path
    max_len_para = int(args.max_len_para)
    max_len_arti = int(args.max_len_arti)
    model_name = args.model_name

    train_model(dataset=dataset, max_len_arti=max_len_arti, max_len_para=max_len_para,
                semantic_dim=semantic_dim, device=device, save_path=save_path,
                store_threshold=store_threshold,
                model_name=model_name)





