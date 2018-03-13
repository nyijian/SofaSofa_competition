import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

#需要进行特殊处理的features
club_index = 1
league_index = 2
birth_date_index = 3
nationality_index = 6
work_rate_att_index = 17
work_rate_def_index = 18
preferred_foot_index = 19

speciel_index = [club_index,league_index,birth_date_index,nationality_index,work_rate_att_index,work_rate_def_index,preferred_foot_index]

#加载csv文件数据
def load_data(filepath):
    data = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)
    data = np.array(data)
    #把空字符串替换为0
    data[data==""]=0
    return data

def data_scaler_train(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    #保存scaler，以便处理test数据集时调用
    output = open('scaler.pkl', 'wb')
    pickle.dump(scaler,output)
    return data_scaled

def data_scaler_test(data):
    #用pickle加载scaler
    input = open('scaler.pkl', 'rb')
    scaler = pickle.load(input)
    data_scaled = scaler.transform(data)
    return data_scaled

def preprocess(data):
    #筛选出不需要进行特殊处理的features
    data_normal = np.delete(data,speciel_index,axis=1)
    data_normal_scaled = data_scaler_train(data_normal)


if __name__ == "__main__":
    filepath = "./data/train.csv"
    data = load_data(filepath)
    preprocess(data)

