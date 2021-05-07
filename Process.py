import pickle
import numpy as np
import os
import math


class Processing:
    def __init__(self, path, save_path, start_year, end_year, data_type):
        self.path = path
        self.year = None
        self.start_year = start_year
        self.end_year = end_year
        self.pic_data = None
        self.save_path = save_path
        self.data_type = data_type
        self.flag = False  # Determine if the currently read folder data is OE
        self.argo_classes = ['Alt purified AE', 'Alt purified CE', 'Outside eddy']
        self.oe_path = None


    def load(self, path):
        """
        load Argo/xgb data
        """
        file_path = path
        with open(file_path, 'rb') as f:
            self.pic_data = pickle.load(f)


    def choose(self):
        """
        Choose the NE data with interpolation method
        """
        self.load(self.oe_path)
        sum_array = np.zeros((2, len(self.pic_data)))
        for j in range(len(self.pic_data)):
            data = self.pic_data[j]
            if self.data_type == 'Argo':
                information = data['pda_profile']
            else:
                information = data[4]
            _sum = 0
            for i in range(20, len(information)):
                _sum = _sum + abs(information[i] - information[i-1])

            sum_array[0, j] = _sum
            sum_array[1, j] = j

        final = sum_array[:, sum_array[0, :].argsort()]
        out_eddies = []
        _, n = final.shape
        length = math.floor(0.2 * n)
        for k in range(length):
            index = int(final[1, k])
            out_eddies.append(self.pic_data[index])

        path = self.save_path + str(self.year) + '.pkl'
        if not os.path.exists(path):
            os.mknod(path)
        with open(path, 'wb') as f:
            pickle.dump(out_eddies, f)

        print(f'Successful: {self.year}')


    def argo_profile(self, eddy_class):
        cur_class_data = []
        cur_class_label = []
        all_class_data = []
        for year in range(self.start_year, self.end_year):
            pkl_path = self.path + self.data_type + '/' + eddy_class + '/' + str(year) + '.pkl'

            if self.flag == True:
                self.year = year
                self.oe_path = pkl_path
                self.load(pkl_path)
                for data in self.pic_data:
                    pda_profile = data['pda_profile']
                    lat = data['argo_lat']
                    lon = data['argo_lon']
                    eddy_year = str(data['date'])[:4]
                    eddy_date = str(data['date'])[4:]
                    all_class_data.append(pda_profile[20:] + [lat, lon, int(eddy_year), int(eddy_date)])

                self.choose()
                eddy_class = 'Alt purified NE'
                pkl_path = self.path + self.data_type + '/' + eddy_class + '/' + str(year) + '.pkl'

            if os.path.exists(pkl_path):
                self.load(pkl_path)
                for data in self.pic_data:
                    pda_profile = data['pda_profile']
                    lat = data['argo_lat']
                    lon = data['argo_lon']
                    eddy_year = str(data['date'])[:4]
                    eddy_date = str(data['date'])[4:]

                    if eddy_class == 'Alt purified AE':
                        cur_class_label.append(0)
                    elif eddy_class == 'Alt purified CE':
                        cur_class_label.append(1)
                    else:
                        cur_class_label.append(2)
                    cur_class_data.append(pda_profile[20:] + [lat, lon, int(eddy_year), int(eddy_date)])

        return cur_class_data, cur_class_label, all_class_data


    def xbt_profile(self, folder_name):
        cur_data = []
        cur_label = []
        all_cur_data = []
        for year in range(self.start_year, self.end_year):
            pkl_path = self.path + self.data_type + '/' + folder_name + '/' + str(year) + '.pkl'

            if self.flag == True:
                self.oe_path = pkl_path
                self.load(pkl_path)
                for data in self.pic_data:
                    temp = data[4][20:]
                    x = np.isnan(temp)
                    if True in x:
                        continue
                    else:
                        all_cur_data.append(data[4][20:] + [data[0], data[1], int(str(data[2])[:4]), int(str(data[2])[4:])])

                self.choose()
                folder_name = 'Alt purified NE'
                pkl_path = self.path + self.data_type + '/' + folder_name + '/' + str(year) + '.pkl'
                if os.path.exists(pkl_path):
                    self.load(pkl_path)
                    for data in self.pic_data:
                        temp = data[4][20:]
                        x = np.isnan(temp)
                        if True in x:
                            continue
                        else:
                            cur_label.append(2)
                            cur_data.append(data[4][20:] + [data[0], data[1], int(str(data[2])[:4]), int(str(data[2])[4:])])

            else:
                if os.path.exists(pkl_path):
                    self.load(pkl_path)
                    for data in self.pic_data:
                        _type = data[6]['sign_type']
                        if _type == 'Anticyclonic':
                            cur_label.append(0)
                        else:
                            cur_label.append(1)

                        cur_data.append(data[4][20:] + [data[0], data[1], int(str(data[2])[:4]), int(str(data[2])[4:])])

        return cur_data, cur_label

    def input(self):
        """
        get the input data for the network
        """
        if self.data_type == 'Argo':
            input_data = []
            input_label = []
            All_data = []
            for eddy_class in self.argo_classes:
                if eddy_class == 'Outside eddy':
                    self.flag = True

                else:
                    self.flag = False

                datas, labels, all_data = self.argo_profile(eddy_class)
                input_data = input_data + datas
                input_label = input_label + labels
                All_data = All_data + all_data

                input_data = norm(All_data, input_data)

        else:
            input_data = []
            input_label = []

            xbt_path = self.path + self.data_type + '/'
            folders = os.listdir(xbt_path)

            for folder in folders:
                if folder == 'XBT_outeddy_abnomal':
                    self.flag = True
                else:
                    self.flag = False

                cur_data, cur_label, all_data = self.xbt_profile(folder)
                input_data = input_data + cur_data
                input_label = input_label + cur_label
                All_data = All_data + all_data

                input_data = norm(All_data, input_data)


        return input_data, input_label


def norm(all_data, input_data):
    all_data = np.array(all_data)
    input_data = np.array(input_data)
    _, n = all_data.shape
    for i in range(n):
        data_max = max(all_data[:, i])
        data_min = min(all_data[:, i])
        input_data[:, i] = (np.array(input_data[:, i]) - data_min) / (data_max - data_min)

    return input_data





