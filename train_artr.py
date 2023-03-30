import torch
from model_components.validator_artr import validate
from model_config.artr import Artr
import argparse
from model_components import dataloader_artr
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from model_components.clustering_loss import LossFunc
from tqdm import tqdm

def train_model(dataset, max_len_arti, semantic_dim, model_name,
                device, save_path, max_len_para, store_threshold):
    LR = 5e-5
    epoch = 10000
    loss_all = []

    train_dataloader = dataloader_artr.get_dataloader(goal='train',
                                                      device=device,
                                                      dataset_name=dataset,
                                                      batch_size=batch_size,
                                                      model_name=model_name)

    test_dataloader = dataloader_artr.get_dataloader(goal='test',
                                                      device=device,
                                                      dataset_name=dataset,
                                                      batch_size=batch_size,
                                                      model_name=model_name)

    model = Artr(num_obj_query=max_len_arti, hidd_dim=semantic_dim, device=device)
    model.to(device=device)
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=2, verbose=True)
    loss_func = LossFunc(device=device, num_query=max_len_arti)
    for epoch_num in range(epoch):
        for data in tqdm(train_dataloader):
            label = torch.flatten(data['label'])
            # print(label)
            text_cls = data['text_cls']
            text_tok = data['text_tok']
            b_s, num_sen, _ = text_cls.shape
            text_cls = text_cls.view(1, b_s*num_sen, semantic_dim)
            text_tok = text_tok.view(1, b_s*num_sen, max_len_para, semantic_dim)
            output = model(text_cls, text_tok)
            query_result = output['query_result']
            cos_sim = output['cos_sim']
            loss = loss_func(cos_sim, label, query_result)
            # print(loss)
            loss_all.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break

        print(epoch_num)
        print(sum(loss_all) / len(loss_all))
        loss_all = []

        val_loss, acc = \
            validate(model=model, dataloader=test_dataloader, max_len_para=max_len_para,
                            loss_func=loss_func, semantic_dim=semantic_dim)
        scheduler.step(val_loss)

        if acc > store_threshold:
            torch.save(model.state_dict(),
                       'model_zoo/'+save_path+str(epoch_num)+'.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='agnews')
    parser.add_argument("--store_threshold", default=0.75)
    parser.add_argument("--save_path", default='agnews')
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--model_type", default='no_con')
    parser.add_argument("--max_len_para", default=256)
    parser.add_argument("--max_len_arti", default=2)
    parser.add_argument("--cluster_num", default=2)
    parser.add_argument("--model_name", default='shalomma/llama-7b-embeddings')
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





