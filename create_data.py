import argparse
import os
from ast import literal_eval
import json
import pandas
import pandas as pd
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
def create_data_cell(path):
    text = []
    path_all = [path+'/'+x for x in os.listdir(path)]
    for path_cell in path_all:
        with open(path_cell, 'r') as file:
            content = file.read()
        text.append(content.replace('\n', ' '))
    random.shuffle(text)
    text_train = text[:int(len(text)*0.7)]
    text_test = text[int(len(text)*0.7):]
    return text_train, text_test
if __name__ == "__main__":
    dataset = load_dataset('yelp_review_full')
    df = dataset['train']
    df = pd.DataFrame(df)
    df.columns = ['Class Index', 'Title']
    df = df[df['Class Index']!=2]
    df.loc[df['Class Index'] < 2, 'Class Index'] = 1
    df.loc[df['Class Index'] > 2, 'Class Index'] = 2
    df = df.dropna(axis=0)
    df['Title'] = df['Title'].map(lambda x: str(x).replace('\\n', ''))
    df.to_csv('data/train_yelp.csv')
    print(dataset)
    # df = pd.read_csv('data/imdb.csv')
    # df_1, df_2 = df.groupby('Class Index')
    # df_1= df_1[1]
    # df_2 = df_2[1]
    # df_1 = df_1.sample(frac=1).reset_index(drop=True)
    # df_2 = df_2.sample(frac=1).reset_index(drop=True)
    # df_1_train, df_1_test = train_test_split(df_1, test_size=0.3)
    # df_2_train, df_2_test = train_test_split(df_2, test_size=0.3)
    # df_train = pd.concat([df_1_train, df_2_train], axis=0)
    # df_train = df_train.sample(frac=1).reset_index(drop=True)
    # df_train.to_csv('data/train_imdb.csv')
    # df_test = pd.concat([df_1_test, df_2_test], axis=0)
    # df_test = df_test.sample(frac=1).reset_index(drop=True)
    # df_test.to_csv('data/test_imdb.csv')
    # df = pd.read_csv('data/IMDB Dataset.csv')
    # df.columns = ['Title', 'Class Index']
    # df = df.dropna(axis=0)
    # dic = {'positive':1, 'negative':2}
    # df['Class Index'] = df['Class Index'].map(lambda x: dic[x])
    # df['Title'] = df['Title'].map(lambda x: str(x).replace('<br /><br />', ''))
    # df.to_csv('data/imdb.csv')
    # df = pandas.read_csv('data/yahootest.csv')
    # df.columns = ['Class Index', 'Title', 'temp', 'Description']
    # df = df.dropna(axis=0)
    # df['Class Index'] = df['Class Index'].map(lambda x: int(x))
    # df['Title'] = df['Title'].map(lambda x: str(x).replace('<br />\\n', '').replace('\\n', ''))
    # df['Description'] = df['Description'].map(lambda x: str(x).replace('<br />\\n', '').replace('\\n', ''))
    # df.to_csv('data/test_yahoo.csv')
    # print(df)
    # text_train = []
    # text_test = []
    # path = 'data/bbc'
    # path_all = os.listdir(path)
    # path_all = ['data/bbc/'+x for x in path_all]
    # print(path_all)
    # for x in path_all:
    #     text_train.append(create_data_cell(x)[0])
    #     text_test.append(create_data_cell(x)[1])
    #
    # index = []
    # for i, sublst in enumerate(text_train):
    #     for item in sublst:
    #         index.append(i+1)
    # text_all = [item for sublst in text_train for item in sublst]
    # df_train = pd.DataFrame({'Class Index':index,'Description': text_all}).sample(frac=1, random_state=42).reset_index(drop=True)
    #
    # index = []
    # for i, sublst in enumerate(text_test):
    #     for item in sublst:
    #         index.append(i+1)
    # text_all = [item for sublst in text_test for item in sublst]
    # df_test = pd.DataFrame({'Class Index': index, 'Description': text_all}).sample(frac=1, random_state=42).reset_index(drop=True)
    # df_train.to_csv('data/train_bbc.csv')
    # df_test.to_csv('data/test_bbc.csv')


