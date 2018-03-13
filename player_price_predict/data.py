import csv
import numpy as np

club_index = 1
league_index = 2
birth_date_index = 3
nationality_index = 6
weak_foot_index = 16
work_rate_att_index = 17
work_rate_def_index = 18
preferred_foot_index = 19
def load_data(filepath):
    data = []
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)
    return np.array[data]

# def preprocess(data):
if __name__ == "__main__":
    with open("./data/train.csv") as f:
        reader = csv.reader(f)
        next(reader)
        print(next(reader))
        a = np.array(next(reader))
        print(a)
        print (a[1]/a[2])
