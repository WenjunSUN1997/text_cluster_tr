from torch.utils.data.dataloader import DataLoader
from ast import literal_eval
from model_components.datasetor_agnews import TextDataSeter
from model_components.datasetor_bbc import TextDataSeterBbc
from model_components.datasetor_yahoo import TextDataSeterYahoo
from model_components.datasetor_clustering import TextDataSeterCluster
def get_dataloader(dataset_name, goal, device, batch_size, model_name):

    file_path_list = {'agnews':{'train': 'data/train.csv',
                                'test': 'data/test.csv'},
                      'bbc':{'train':'data/train_bbc.csv',
                             'test':'data/test_bbc.csv' },
                      'yahoo':{'train':'data/train_yahoo.csv',
                             'test':'data/test_yahoo.csv' },
                      'imdb': {'train': 'data/train_imdb.csv',
                                'test': 'data/test_imdb.csv'},
                      'yelp': {'train': 'data/train_yelp.csv',
                               'test': 'data/test_yelp.csv'},
                      }

    datasetor_list = {'agnews': TextDataSeter,
                      'bbc': TextDataSeterBbc,
                      'yahoo': TextDataSeterCluster,
                      'imdb':TextDataSeterCluster,
                      'yelp':TextDataSeterCluster,}

    dataset = TextDataSeterCluster(file_path_list[dataset_name][goal], device, model_name)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

