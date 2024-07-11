import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd


class DATA_CLEAN():
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.dataset = []  # 用于存储30个number的数据
        self.dataset_mean = []  # 用于存储30个number的均值
        self.dataset_clean_after_row = []  # 存储经过 row 清洗之后的数据
        self.dataset_clean_scores = []  # 存储所有分数
        self.dataset_clean_after_column = []  # 存储经过 column 清洗之后的数据
        self.dataset_clean_vary = []  # 存储清洗之后的变化值
        self.dataset_mean_vary = []  # 存储原始数据的变化值

    def loading_raw_data(self):
        for number_data in self.raw_data:
            number = []
            number_data = np.array(number_data)
            continuous_exception_flag = 0
            for i in range(1, 26):
                number_length_data = []
                for item in number_data:
                    if item[0] == i:
                        number_length_data.append(item[1])
                if len(number_length_data) == 0:
                    continuous_exception_flag += 1
                    if i == 1:  # 第一段没有数据
                        number_length_data.append([-95])
                    elif continuous_exception_flag >= 3 and np.round(np.mean(number[-1]), 2) < -80:  # 连续三段以上没有数据
                        number_length_data.append([-95])
                    elif np.round(np.mean(number[-1]), 2) > -85:  # 非第一段没有数据，且上一段的均值大于-85
                        number_length_data.append(number[-1])
                    else:  # 非第一段没有数据，且上一段的均值小于-85
                        number_length_data.append([-95])
                    number.append(*number_length_data)
                else:
                    continuous_exception_flag = 0
                    number.append(number_length_data)
            self.dataset.append(number)

    # 创建一个孤立森林，对数据做出异常评分，同时将 scores  data传给 self.processing_row_data，最后返回一个处理之后的结果
    def isolated_forest(self, data, n_estimators, max_samples, contamination, max_feature, row=True):
        iforest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination,
                                  max_features=max_feature, bootstrap=False, n_jobs=-1, random_state=1)
        iforest.fit(data[:, -1].reshape(-1, 1))  # 训练
        scores = iforest.decision_function(data[:, -1].reshape(-1, 1))  # 返回异常分数
        if row:
            return self.processing_row_data(scores, data[:, -1]), scores
        else:
            return self.processing_column_data(scores, data)

    @staticmethod
    def processing_row_data(scores, data):
        data_length = []  # 临时列表，用于返回一个 number， 一个 length 对应处理之后的数据
        for index, score in enumerate(scores):
            if score <= -0.1:
                data_length.append([1, data[index]])
            else:
                data_length.append([0, data[index]])
        return data_length

    @staticmethod
    def processing_column_data(scores, data):
        data_column = []
        for index, score in enumerate(scores):
            if score <= -0.1:
                flag = 2 + data[index][0]
            else:
                flag = 0 + data[index][0]
            data_column.append([flag, data[index][1]])
        return data_column

    def data_return_after_row_cleaned(self):
        row_cleaned_data = []
        for number_data in self.dataset_clean_after_row:
            mean_number = []
            for number_length_data in number_data:
                if len(number_length_data) == 0:
                    mean_number.append(mean_number[-1])
                    continue
                clean_data = [item[1] for item in number_length_data if item[0] == 0]
                mean_number.append(np.round(np.mean(clean_data), 2))
            row_cleaned_data.append(mean_number)

        return row_cleaned_data

    def data_return_after_column_cleaned(self):
        column_cleaned_data = []
        for number_data in self.dataset_clean_after_column:
            clean_data = [item[1] for item in number_data if item[0] < 2]
            column_cleaned_data.append(np.round(np.mean(clean_data), 2))
        return column_cleaned_data

    # 循环处理30个 number 25个 length 的数据
    def start_clean(self):
        # 横向
        for num in range(len(self.dataset)):
            data_num_cleaned = []
            data_num_cleaned_score = []
            for length in range(len(self.dataset[0])):
                if len(self.dataset[num][length]) <= 4:
                    data_num_cleaned.append([[0, item] for item in self.dataset[num][length]])  # 假设当前无异常值
                    continue
                if 4 < len(self.dataset[num][length]) <= 8:
                    contamination = 0.05  # 只有一个异常值
                if 8 < len(self.dataset[num][length]) <= 15:
                    contamination = 0.12  # 只有两个异常值
                else:
                    contamination = 0.15  # 三个异常值以上
                data_length_clean = self.isolated_forest(np.array(self.dataset[num][length]).reshape(-1, 1),
                                                         30, len(self.dataset[num][length]), contamination, 1)
                data_num_cleaned.append(data_length_clean[0])
                data_num_cleaned_score.append(data_length_clean[1])
            self.dataset_clean_after_row.append(data_num_cleaned)
            self.dataset_clean_scores.append(data_num_cleaned_score)
        # 纵向
        for i in range(len(self.dataset_clean_after_row[1])):
            data = [sub for sublist in self.dataset_clean_after_row for sub in sublist[i][:]]  # 拿到所有 number 下的所有 length
            data_column_clean = self.isolated_forest(np.array(data).reshape(-1, 2),
                                                     100, len(data), 0.1, 1, row=False)
            self.dataset_clean_after_column.append(data_column_clean)


if __name__ == '__main__':
    random_array = []
    # df = pd.read_csv('C:/Users/42917/Desktop/导出数据/data_clean{}_{}.csv'.format(1, 9))
    # for num in range(1, 31):  # 判断number是第几次
    #     data_num = df[df["number"] == num]
    #     data_num_item = []
    #     for item in data_num.values:
    #         data_num_item.append([item[0], item[1]])
    #     random_array.append(data_num_item)
    # # print(random_array)
    # """#######################把random_array替换为对应的数组：30*n*2##################"""
    # data_clean = DATA_CLEAN(random_array)
    # data_clean.loading_raw_data()
    # data_clean.start_clean()
    #
    # """最后的数据"""
    # data_after_row = data_clean.data_return_after_row_cleaned()  # 30*25
    #
    # data_after_column = data_clean.data_return_after_column_cleaned()  # 1*25
    # print(data_after_column)
    # print(data_after_row)
    # print("############")
