import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier

def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def imread_binary(path):
    img = cv2.imread(path, 0)
    ret, bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return(bin)

class PWSegmentation():

    def __init__(self, img_rep, bin_rep_list, class_names, resize, model=None):
        self.resize = resize
        self.img_names = np.asarray(os.listdir(img_rep))

        is_complete = np.repeat(True, self.img_names.shape[0])
        for bin_rep in bin_rep_list:
            bin_img_list = np.asarray(os.listdir(bin_rep))
            is_complete = is_complete & np.isin(self.img_names, bin_img_list)

        self.img_names = self.img_names[is_complete]
        self.class_names = class_names
        self.img_rep = img_rep
        self.bin_rep_list = bin_rep_list
        self.data = None
        self.model = model

        print("Valid obs:")
        print(self.img_names)
        print("Class names")
        print(class_names)

    def concat_data(self):
        img_list = [cv2.resize(cv2.imread(self.img_rep + "/" + img_name), (0, 0), fx=self.resize, fy=self.resize)
                    for img_name in self.img_names]

        tab = []

        for i in range(len(self.img_names)):
            mask_list = [((cv2.resize(imread_binary(bin_rep + "/" + self.img_names[i]), (img_list[i].shape[1], img_list[i].shape[0]), cv2.INTER_NEAREST))).reshape(-1)
                         for bin_rep in self.bin_rep_list]

            annotate = np.repeat(self.class_names[0], len(mask_list[0]))
            for j in range(len(mask_list)):
                annotate[mask_list[j] > 100] = self.class_names[j]

            img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2HSV)
            hue = img_list[i][:, :, 0].reshape(-1)
            saturation = img_list[i][:, :, 1].reshape(-1)
            value = img_list[i][:, :, 2].reshape(-1)
            posx = (np.array([np.arange(img_list[i].shape[0]), ] * img_list[i].shape[1]).transpose() - img_list[i].shape[0] / 2).reshape(-1)
            posy = (np.array([np.arange(img_list[i].shape[1]), ] * img_list[i].shape[0]) - img_list[i].shape[1] / 2).reshape(-1)

            tab.append(np.vstack([annotate, hue, saturation, value, posx, posy]).transpose())

        self.data = np.vstack(tab)

    def train(self, ntrees=20, max_depth=10):
        if self.data is None:
            self.concat_data()
        X = self.data[:, np.arange(6) != 0]
        y = self.data[:, 0]
        self.model = RandomForestClassifier(n_estimators=ntrees, max_depth=max_depth, random_state=0, oob_score=True)
        self.model.fit(X, y)
        print("OOB Score:/n")
        print(self.model.oob_score_ )

    def predict(self, img, display=True):
        if self.train is None:
            self.train()

        img_hsv = cv2.cvtColor(cv2.resize(img, (0, 0), fx=self.resize, fy=self.resize), cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0].reshape(-1)
        saturation = img_hsv[:, :, 1].reshape(-1)
        value = img_hsv[:, :, 2].reshape(-1)
        posx = (np.array([np.arange(img_hsv.shape[0]), ] * img_hsv.shape[1]).transpose() - img_hsv.shape[0] / 2).reshape(-1)
        posy = (np.array([np.arange(img_hsv.shape[1]), ] * img_hsv.shape[0]) - img_hsv.shape[1] / 2).reshape(-1)

        X = np.vstack([ hue, saturation, value, posx, posy]).transpose()
        y = self.model.predict(X)

        bin_img_predict = [cv2.cvtColor(((y == name).astype('uint8') * 255).reshape(img_hsv.shape[0:2]), cv2.COLOR_GRAY2BGR) for name in self.class_names]
        bin_img_resize = [cv2.resize(bin, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST) for bin in bin_img_predict]

        result = np.zeros(bin_img_predict[0].shape).astype('uint8')
        colors = get_spaced_colors(len(self.class_names)+1)

        for i in range(len(self.class_names)):
            result[np.where((bin_img_predict[i] == [255, 255, 255]).all(axis=2))] = list(colors[i+1])

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = cv2.resize(result, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)

        if display:
            cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("test", result)
            cv2.waitKey(0)

        return(bin_img_resize, result)





