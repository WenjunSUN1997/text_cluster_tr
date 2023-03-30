import pandas as pd
from sklearn.utils import shuffle
from .dataset_creator import TextDataSeter
from torch.utils.data import DataLoader

def get_dataloader(goal, lang, data_type, batch_size, device):
    data_type_dict = {'random': '_text_bbox_data_',
                      'one_by': '_text_bbox_data_one_by_'}
    csv = pd.read_csv('train_test_data/'+ goal + data_type_dict[data_type]+ lang + '.csv')
    if goal == 'train':
        csv_1 = csv[csv['label'] == 1]
        csv_0 = csv[csv['label'] == 0]
        length = min(len(csv_1['label']), len(csv_0['label']))
        index_1 = shuffle(list(range(len(csv_1))))[:length]
        index_0 = shuffle(list(range(len(csv_0))))[:length]
        csv_1 = csv_1.iloc[index_1].reset_index(drop=True)
        csv_0 = csv_0.iloc[index_0].reset_index(drop=True)
        csv = pd.concat([csv_0, csv_1])

    index = list(range(len(csv)))
    index = shuffle(index)
    csv = csv.iloc[index].reset_index(drop=True)
    dataset = TextDataSeter(csv, lang, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader