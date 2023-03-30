import torch
from tqdm import tqdm

def val_model(model, test_dataloader, loss_function, device, lang):
    print('validating........')
    loss_list = []
    label_all = 0
    n_correct = 0
    step = 0

    for data in tqdm(test_dataloader):
        result = []
        text_1 = data['text_1']
        text_2 = data['text_2']
        text_all = data['text_all']
        label = data['label'].to(device)
        x_1 = data['x_1'].to(device)
        y_1 = data['y_1'].to(device)
        x_2 = data['x_2'].to(device)
        y_2 = data['y_2'].to(device)
        text_1_cls, text_2_cls, semantic_1_cls, semantic_2_cls = \
            model(text_1, text_2, text_all, x_1, y_1, x_2, y_2)
        loss = loss_function(text_1_cls, text_2_cls,
                             semantic_1_cls, semantic_2_cls, label)
        loss_list.append(loss.item())
        sim = torch.cosine_similarity(semantic_1_cls, semantic_2_cls, dim=-1)
        for x in sim:
            if x < 0.2:
                result.append(-1)
            else:
                result.append(1)

        n_correct += (torch.tensor(result).to(device) == label).sum().item()
        label_all += len(label)
        step += 1
        if lang=='fre':
            if step > 3000:
                break

    print('val loss:', sum(loss_list) / len(loss_list))
    print('val acc:', n_correct / label_all)