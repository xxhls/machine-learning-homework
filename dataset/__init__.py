import cv2
import os
import numpy as np


def read_files(file_path):
    result = os.listdir(file_path)
    for i in range(len(result)):
        result[i] = file_path + "/" + result[i]
    return result


def read_img(file_path):
    return cv2.imread(file_path)


def split_img(file_path):
    _img = read_img(file_path)
    _tmp_list = []
    _row = 29
    _col = 50
    # 切片29x50
    while _row < len(_img):
        while _col < len(_img[0]):
            _piece = _img[_row - 29: _row, _col - 50: _col]
            _tmp_list.append(_piece)
            _col += 50
        _row += 29
    return np.array(_tmp_list)


if __name__ == "__main__":
    res = read_files("./bobi")
    data_bobi = []
    for i in range(len(res)):
        data = split_img(res[i])
        for j in range(len(data)):
            data_bobi.append(data[j])
    data_bobi = np.array(data_bobi)
    # 0 为波比
    label_bobi = np.zeros(len(data_bobi))
    print(data_bobi.shape)
    print(label_bobi.shape)
    res = read_files("./xiaopao")
    data_xiaopao = []
    for i in range(len(res)):
        data = split_img(res[i])
        for j in range(len(data)):
            data_xiaopao.append(data[j])
    data_xiaopao = np.array(data_xiaopao)
    # 1 为小炮
    label_xiaopao = np.ones(len(data_xiaopao))
    print(data_xiaopao.shape)
    print(label_xiaopao.shape)
    # 按照7:3划分测试集和训练集
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(int(len(data_bobi) * 0.7)):
        train_images.append(data_bobi[i])
        train_labels.append(label_bobi[i])
    for i in range(int(len(data_bobi) * 0.7), len(data_bobi)):
        test_images.append(data_bobi[i])
        test_labels.append(label_bobi[i])
    for i in range(int(len(data_xiaopao) * 0.7)):
        train_images.append(data_xiaopao[i])
        train_labels.append(label_xiaopao[i])
    for i in range(int(len(data_xiaopao) * 0.7), len(data_xiaopao)):
        test_images.append(data_xiaopao[i])
        test_labels.append(label_xiaopao[i])
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    np.savez("dataset",
             train_images=train_images,
             train_labels=train_labels,
             test_images=test_images,
             test_labels=test_labels
             )
