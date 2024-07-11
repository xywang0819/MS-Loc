# -*- coding: utf-8 -*-
import os.path
import logging
import numpy as np
import torch
import random
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
from RBM import RBM
import matplotlib.pyplot as plt
import pandas as pd


class DBN:
    def __init__(self, input_size, layers, mode='bernoulli', gpu=False, k=5, savefile=None, cuda=None):
        self.layers = layers
        self.input_size = input_size
        self.layer_parameters = [{'W': None, 'hb': None, 'vb': None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile
        self.dev = cuda

    def sample_v(self, y, W, vb):
        wy = torch.mm(y, W)
        activation = wy + vb
        p_v_given_h = torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        else:
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))

    def sample_h(self, x, W, hb):
        wx = torch.mm(x, W.t())
        activation = wx + hb
        p_h_given_v = torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_h_given_v, torch.bernoulli(p_h_given_v)  # 伯努利实验
        else:
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

    def generate_input_for_layer(self, index, x):
        if index > 0:
            x_gen = []
            for _ in range(self.k):
                x_dash = x.clone()
                for i in range(index):
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
                x_gen.append(x_dash)

            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
        else:
            x_dash = x.clone()
        return x_dash

    def train_DBN(self, x):
        for index, layer in enumerate(self.layers):
            if index == 0:
                vn = self.input_size
            else:
                vn = self.layers[index - 1]
            hn = self.layers[index]

            rbm = RBM(vn, hn, epochs=100, mode='bernoulli', lr=0.0004, k=20, batch_size=90,
                      gpu=True, optimizer='adam', early_stopping_patience=5, cuda=self.dev)
            x_dash = self.generate_input_for_layer(index, x)
            rbm.train(x_dash)
            self.layer_parameters[index]['W'] = rbm.W.cpu()
            self.layer_parameters[index]['hb'] = rbm.hb.cpu()
            self.layer_parameters[index]['vb'] = rbm.vb.cpu()
            print("Finished Training Layer:", index, "to", index + 1)
        if self.savefile is not None:
            torch.save(self.layer_parameters, self.savefile)

    def reconstructor(self, x):
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(len(self.layer_parameters)):
                x_dash, _ = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(len(self.layer_parameters)):
                i = len(self.layer_parameters) - 1 - i
                y_dash, _ = self.sample_v(y_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['vb'])
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)
        return y_dash, x_dash

    def initialize_model(self):
        # print("The Last layer will not be activated. The rest are activated using the Sigoid Function")
        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer['W'].shape[1], layer['W'].shape[0]))
            if index < len(self.layer_parameters) - 1:  # 判断是否非最后一层
                modules.append(torch.nn.Sigmoid())
        model = torch.nn.Sequential(*modules)

        for layer_no, layer in enumerate(model):
            if layer_no // 2 == len(self.layer_parameters) - 1:
                break
            if layer_no % 2 == 0:
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no // 2]['W'])
                model[layer_no].bias = torch.nn.Parameter(self.layer_parameters[layer_no // 2]['hb'])

        return model

    def load_state_dict(self):
        for index, layer in enumerate(self.layers):
            loaded = torch.load(self.savefile)
            self.layer_parameters[index]['W'] = loaded[index]["W"]
            self.layer_parameters[index]['hb'] = loaded[index]['hb']
            self.layer_parameters[index]['vb'] = loaded[index]['vb']

def add_gaussian_noise(data, mean, std):
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return np.vstack((data, noisy_data))


def min_max_inverse(normalized_data, min_value, max_value):
    return normalized_data * (max_value[:, np.newaxis] - min_value[:, np.newaxis]) + min_value[:, np.newaxis]


# 将模型跑完之后重构的数据保存
def save2csvfile(path, *args):
    if os.path.exists(path):
        df0 = pd.read_csv(path)
    else:
        df0 = pd.DataFrame()
    combined_data = pd.concat([df0] + list(args), axis=1)  # 使用 *args 合并数据
    combined_data.to_csv(path, index=False)  # 保存合并后的数据到指定的文件


def task(cuda, number):
    try:
        for _ in range(38, 43):
            if _ == 13 or _ == 8 or _ == 23 or _ == 30:
                continue
            dataset = []
            df = pd.read_csv("./dataset/data/Data_after_the_isolated_forest/length_rss{}_{}.csv".format(_, number))
            for num in range(1, 31):
                data = df[df["number"] == num]
                dataset.append(data['rss_vary'])
            dataset = np.array(dataset, dtype=np.float32)[5:]
            # 添加噪声扩充数据
            for i in range(9):
                dataset = add_gaussian_noise(dataset, 0, i / 10)

            # 最大最小归一化
            dataset_max = dataset.max(axis=1)
            dataset_min = dataset.min(axis=1)

            dataset = np.array(dataset, dtype=np.float32)
            epsilon = 1e-10  # 防止在最大最小归一化的时候出现，分母为0的情况
            dataset_Normalization = (dataset - dataset.min(axis=1)[:, np.newaxis]) / (dataset.max(axis=1) - dataset.min(axis=1) + epsilon)[:, np.newaxis]
            dataset_Normalization = torch.from_numpy(dataset_Normalization)

            layers = [500, 350, 150, 50]
            dbn = DBN(25, layers, savefile="./Model_Storage/4_floor_25/model/model_{}_{}".format(_, number), cuda=cuda)
            print("数据 {}_{} 的模型正在训练,使用的是{}".format(_, number, cuda))
            dbn.train_DBN(dataset_Normalization)
            dbn.initialize_model()
            y = dbn.reconstructor(dataset_Normalization)  # 重构回来的数据

            y_reduction = min_max_inverse(y[0], dataset_min, dataset_max)  # 将数据拉回原数据

            print('\n\n\n')
            print("MAE of an all 0 reconstructor:", torch.mean(dataset_Normalization).item())
            print("MAE between reconstructed and original sample:",
                  torch.mean(torch.abs(y_reduction[:30, :] - dataset_Normalization[:30, :])).item())

            plt.figure(figsize=(24, 16))
            title = "Pic from csv {}_{}".format(_, number)
            plt.suptitle(title, fontsize=24)
            plt.subplot(2, 2, 1)
            plt.title("Origin", fontsize=22)
            for i, line_data in enumerate(dataset[:30, :]):
                plt.plot(range(1, 26), line_data, label=f'Line {i + 1}')

            plt.subplot(2, 2, 2)
            plt.title("Normalization", fontsize=22)
            for i, line_data in enumerate(dataset_Normalization[:30, :]):
                plt.plot(range(1, 26), line_data, label=f'Line {i + 1}')

            plt.subplot(2, 2, 3)
            plt.title("Reconstruction_Origin", fontsize=22)
            for i, line_data in enumerate(y_reduction[:30, :]):
                plt.plot(range(1, 26), line_data, label=f'Line {i + 1}')

            plt.subplot(2, 2, 4)
            plt.title("Reconstruction_Normalization", fontsize=22)
            for i, line_data in enumerate(y[0][:30, :]):
                plt.plot(range(1, 26), line_data, label=f'Line {i + 1}')

            plt.tight_layout()
            plt.savefig(r"./Model_Storage/4_floor_25/Pic/Pic_from_csv_{}_{}.png".format(_, number))
            plt.close()
    except Exception as e:
        logging.error(f"Error in getting {number} result: {e}")


if __name__ == '__main__':
    task("cuda:0", 14)  # 1-24 跑到 了17
    # with ThreadPoolExecutor() as executor:
    #     task_queue = [["cuda:0", 13], ["cuda:1", 14]]
    #     for tasks in task_queue:
    #         executor.submit(task, *tasks)
    #     executor.shutdown()
    # logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    # task_queue_hub = [[["cuda:0", 15], ["cuda:1", 16]],
    #                   [["cuda:0", 17], ["cuda:1", 18]], [["cuda:0", 19], ["cuda:1", 20]]]
    # for task_queue in task_queue_hub:
    #     with ThreadPoolExecutor() as executor:
    #         for tasks in task_queue:
    #             executor.submit(task, *tasks)
    #         executor.shutdown()

