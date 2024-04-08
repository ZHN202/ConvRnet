import numpy as np
import torch
from torch.utils.data import Dataset


# dataSet
class myDataSet(Dataset):
    def __init__(self, dataProcess='MS', train=True, vaild=False, extra=False, path='dataset/'):
        self.dataProcess = dataProcess
        self.train = train
        self.vaild = vaild
        self.extra = extra
        self.path = path
        dataset = []
        if train:
            with open(path + 'dataset_' + self.dataProcess + '.txt') as f:
                for line in f.readlines():
                    data = [float(i) for i in line.split()]
                    dataset.append(data)
            dataset = np.array(dataset)
            data_pics = np.zeros(shape=[17, 3, 51, 51])
            i = 0
            cnt = 0
            for data in dataset:

                cnt += 1
                data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                if cnt == 2601:
                    cnt = 0
                    i += 1
            print(data_pics.shape)
            self.data_pics = data_pics

            np.random.shuffle(dataset)
            self.data = dataset

            del dataset, data_pics
        else:
            if vaild:
                with open(path + 'dataset_' + self.dataProcess + '.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        # if data[2] == 75 or data[2] == 80:
                        #    continue
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:

                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                self.data_pics = data_pics
                dataset = []
                with open(path + 'dataset_val_' + self.dataProcess + '.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                self.data = dataset
            else:

                with open(path + 'dataset_val_' + self.dataProcess + '.txt') as f:
                    for line in f.readlines():
                        data = [float(i) for i in line.split()]
                        if data[2] == 47.5 and not self.extra:
                            break
                        dataset.append(data)
                dataset = np.array(dataset)
                data_pics = np.zeros(shape=[17, 3, 51, 51])
                i = 0
                cnt = 0
                for data in dataset:
                    cnt += 1
                    data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
                    if cnt == 2601:
                        cnt = 0
                        i += 1
                print(data_pics.shape)
                self.data_pics = data_pics
                self.data = dataset
            del dataset, data_pics

    def __getitem__(self, index):

        if self.train:

            # 循环遍历17张数据图
            for i in range(self.data_pics.shape[0]):

                if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
                    return torch.FloatTensor(self.data_pics[i]), \
                        torch.FloatTensor(self.data[index][:3]), \
                        torch.FloatTensor(self.data[index][3:5])
        else:
            if self.vaild:
                # 将角度映射到存在数据图的角度
                angle = self.data[index][2]
                angle_low = (angle // 5) * 5
                angle_high = angle_low + 5
                for i in range(self.data_pics.shape[0]):
                    if self.data_pics[i, -1, 0, 0] == angle_low:
                        pic_low = self.data_pics[i]
                    if self.data_pics[i, -1, 0, 0] == angle_high:
                        pic_high = self.data_pics[i]
                return torch.FloatTensor(((pic_high - pic_low) / 5) * (angle - angle_low) + pic_low), \
                    torch.FloatTensor(self.data[index][:3]), \
                    torch.FloatTensor(self.data[index][3:5])
                # if 5 - (angle - (angle // 5) * 5) < 2.5:
                #     angle = (angle // 5) * 5
                # else:
                #     angle = (angle // 5 + 1) * 5
                # print('使用角度为' + str(angle) + '的数据图')
                # 循环遍历17张数据图
                # for i in range(17):
                #     if self.data_pics[i, -1, 0, 0] == angle:
                #         return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(self.data[index][:3]), torch.FloatTensor(self.data[index][3:5])
            else:
                # 循环遍历数据图
                for i in range(self.data_pics.shape[0]):
                    if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
                        return torch.FloatTensor(self.data_pics[i]), torch.FloatTensor(
                            self.data[index][:3]), torch.FloatTensor(self.data[index][3:5])

    def __len__(self):
        return len(self.data)


class DataSet_moni(Dataset):
    # 1.414971179848037 1.0749577528938408   mean1 mean2
    # 0.3340214520654514 0.20166881182130206 std1 std2
    # 训练集      测试集      是否使用轴间夹角为47.5度的测试集
    def __init__(self, path=''):

        dataset = []

        with open('dataset/dataset_MS.txt') as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]
                # if data[2] == 75 or data[2] == 80:
                #    continue
                data[2] /= 80

                dataset.append(data)

        with open('dataset/dataset_val_MS.txt') as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]
                data[2] /= 80
                dataset.append(data)

        dataset = np.array(dataset)
        data_pics = np.zeros(shape=[19, 3, 51, 51])
        i = 0
        cnt = 0
        for data in dataset:

            cnt += 1
            data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
            if cnt == 2601:
                cnt = 0
                i += 1

        self.data_pics = data_pics

        dataset = []

        with open(path) as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]
                #data[0], data[1] = data[1], data[0]
                data[0] = (data[0] - 1.414971179848037) / 0.3340214520654514
                data[1] = (data[1] - 1.0749577528938408) / 0.20166881182130206

                data[2] /= 80
                dataset.append(data)

        dataset = np.array(dataset)

        # np.random.shuffle(dataset)
        self.data = dataset

        del dataset, data_pics

    def __getitem__(self, index):
        # # 循环遍历17张数据图
        # for i in range(19):
        #     if self.data_pics[i, -1, 0, 0] == self.data[index][2]:
        #         #self.data[index][2] /= 80
        #         return torch.FloatTensor(self.data_pics[i]), \
        #             torch.FloatTensor(self.data[index][:3])

        # 将角度映射到存在数据图的角度
        angle = self.data[index][2] * 80
        angle_low = (angle // 5) * 5 if angle<=80 else 80
        angle_high = angle_low + 5 if angle<=80 else 80
        #print(angle)
        for i in range(self.data_pics.shape[0]):
            if self.data_pics[i, -1, 0, 0] * 80 == angle_low:
                pic_low = self.data_pics[i]
            if self.data_pics[i, -1, 0, 0] * 80 == angle_high:
                pic_high = self.data_pics[i]
        return torch.FloatTensor(((pic_high - pic_low) / 5) * (angle - angle_low) + pic_low), \
            torch.FloatTensor(self.data[index][:3])

    def __len__(self):
        return len(self.data)


class myDataSet_k_fold(Dataset):
    def __init__(self, dataProcess='', path=''):
        self.dataProcess = dataProcess
        dataset = []

        with open('dataset/dataset_' + self.dataProcess + '.txt') as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]
                data[2] /= 80
                dataset.append(data)

        with open('dataset/dataset_val_' + self.dataProcess + '.txt') as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]
                data[2] /= 80
                dataset.append(data)

        dataset = np.array(dataset)
        data_pics = np.zeros(shape=[19, 3, 51, 51])
        i = 0
        cnt = 0
        for data in dataset:

            cnt += 1
            data_pics[i, :, int(data[-1]), int(data[-2])] = data[:3]
            if cnt == 2601:
                cnt = 0
                i += 1

        self.data_pics = data_pics

        dataset = []

        with open(path) as f:
            for line in f.readlines():
                data = [float(i) for i in line.split()]

                data[2] /= 80
                data[3], data[4] = data[3] / 50, data[4] / 50
                dataset.append(data)

        dataset = np.array(dataset)
        print(dataset.shape)

        self.data = dataset

        del dataset, data_pics

    def __getitem__(self, index):

        # loop through 17 data graphs
        for i in range(19):

            if self.data_pics[i, -1, 0, 0] == self.data[index][2]:

                return torch.FloatTensor(self.data_pics[i]), \
                    torch.FloatTensor(self.data[index][:3]), \
                    torch.FloatTensor(self.data[index][3:5])

    def __len__(self):
        return len(self.data)



