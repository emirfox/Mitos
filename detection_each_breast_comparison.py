import random
import statistics

import numpy.linalg.linalg
import pandas as pd
import os

import seaborn
import skrf as rf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from sklearn import neural_network
from sklearn import neighbors
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math
import scipy
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from scipy import stats
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Enter path of the most recent patient list!
path_patient_new = "C:/Users/yurts/Downloads/MMT_15_dec_Onur.xlsx"
#latest

# Enter the path of the data input directory!
path_data_input = "C:/Users/yurts/Documents/MITOS/MITOS/data_update/files_update/"


# Enter the path of the features!

path_feature = "C:/Users/yurts/Documents/MITOS/MITOS/data_frame_features.xlsx"


def reader_(path_patient_new):
    # Let me read the data
    data_ = pd.read_excel(path_patient_new)

    # Let me take only the ones which have biopsy, also, the ones either malignant or benign
    data_ = data_[data_["BX_SCR"] == "bx"]
    data_ = data_[(data_["Pathology"] == "malignant") | (data_["Pathology"] == "benign")]
    data_.reset_index(inplace=True)
    data_ = data_[data_.columns[1:]]
    print("Number of benign patient = {}".format(len(data_[data_["Pathology"] == "benign"])))
    print("Number of malignant patient = {}".format(len(data_[data_["Pathology"] == "malignant"])))

    return data_


data_ = reader_(path_patient_new)

# print(data_)
# print(len(data_))
data_.to_excel("C:/Users/yurts/OneDrive/MITOS/MITOS/data_.xlsx")


def paths_taker(data_, path_data_input):
    # Let me get the s-parameters data!

    all_dir_ = os.listdir(path_data_input)
    all_data_ = []
    for each in all_dir_:
        sub_dirs = os.listdir(path_data_input + each)
        path_all = [path_data_input + each + "/" + i for i in sub_dirs]
        all_data_ = all_data_ + path_all
    check_data = [i.lower() for i in all_data_]
    data_["PatientName"] = [i.lower() for i in data_["PatientName"]]

    # print(data_)
    # print(check_data)

    my_directories = []
    counter = 0
    for each in check_data:
        for every in data_["PatientName"].to_list():
            if every in each:
                my_directories.append(all_data_[counter])
        counter = counter + 1
    # print(len(my_directories))
    # print(my_directories)

    my_paths = []

    for each in my_directories:
        print(os.listdir(each))
        if os.listdir(each)[0] == "temp.s2p" and len(os.listdir(each)) == 1:
            continue
        elif os.listdir(each)[0] == "temp.s2p" and len(os.listdir(each)) > 1:
            my_paths.append(each + "/" + os.listdir(each)[1])
        else:
            my_paths.append(each + "/" + os.listdir(each)[0])

    # print(len(my_paths))
    # print(my_paths)

    return my_paths


my_paths = paths_taker(data_, path_data_input)

# print(len(my_paths))
# print(my_paths)
print("Number of healthy samples = {}".format(len(my_paths) - len(data_)))


# pd.Series(my_paths).to_excel("C:/Users/yurts/OneDrive/MITOS/MITOS/paths_.xlsx")
'''
def path_to_file(my_paths, data_):
    # Let me put the data into data frame and extract both names_surnames as well as the left and right sides!

    # print(my_paths)
    name_second_name = []
    for j in my_paths:
        a = os.path.split(j)[1]
        each = "_".join(a.split("_")[:-5])
        each = each.lower()
        name_second_name.append(each)

    breast_side = []
    for j in my_paths:
        a = os.path.split(j)[1]
        each = a.split("_")[-5]
        if each.lower() == "left":
            breast_side.append("l")

        else:
            breast_side.append("r")

    # print(name_second_name)
    # print(breast_side)
    data_path_ = pd.DataFrame({"PatientName": name_second_name, "Side": breast_side, "paths": my_paths})
    # print(data_path_)
    # print(len(data_path_))
    # print(data_path_["Side"])
    # print(data_["Side"])
    data_path_.to_excel("C:/Users/yurts/OneDrive/MITOS/MITOS/data_path.xlsx")
    merged_ = pd.merge(data_path_, data_, on=["PatientName", "Side"], how="outer")
    merged_.to_excel("C:/Users/yurts/OneDrive/MITOS/MITOS/merged_.xlsx")

    merged_ = merged_.dropna(subset=["paths"])
    merged_.reset_index(inplace=True)
    merged_ = merged_[merged_.columns[1:]]

    merged_.drop_duplicates(inplace=True, keep="first")

    merged_["Pathology"] = merged_["Pathology"].fillna("Healthy")
    # merged_ = merged_[merged_["Pathology"]!="Healthy"]
    merged_.reset_index(inplace=True)
    merged_ = merged_[merged_.columns[1:]]
    data_feature = merged_
    data_feature['Age'] = data_feature['Age'].fillna(data_feature.groupby('PatientName')['Age'].transform('mean'))
    # print(data_feature["Age"])

    data_un = merged_[merged_["PatientName"].isin(data_["PatientName"])]
    # print(data_un)

    return data_feature


data_feature = path_to_file(my_paths, data_)


print(data_feature)

# data_feature.to_excel("C:/Users/yurts/OneDrive/MITOS/MITOS/data_.xlsx")

# # print(len(data_feature))
# #
# #
# # #Now, let me get the parameters!
# # my_list_ = []
# # #
# # #
# # #
def dataframe_create_with_s_parameters_sum(data_feature):
    # data_feature = data_feature[data_feature["Pathology"] != "Healthy"]
    data_feature.reset_index(inplace=True)
    data_feature = data_feature[data_feature.columns[1:]]
    # print(len(data_feature))

    data_feature.loc[
        (data_feature["Pathology"] == "benign") | (data_feature["Pathology"] == "malignant"), "label"] = "Nonhealthy"
    data_feature.loc[(data_feature["Pathology"] == "Healthy"), "label"] = "Healthy"
    data_feature["Pathology"] = data_feature["label"]
    data_feature = data_feature.drop(columns=["label"], axis=1)
    # print(data_feature.columns)
    # print(data_feature["Pathology"])
    data_benign = data_feature[data_feature["Pathology"] == "benign"]
    data_malignant = data_feature[data_feature["Pathology"] == "malignant"]

    my_paths_b = data_feature["paths"].to_list()
    # print(my_paths_b)

    # print(data_feature["paths"].to_list())

    frequencies_b = []
    parameters_b = []
    for each in my_paths_b:
        ntwrk = rf.Network(each)
        frequency_vector = ntwrk.frequency.f
        s_parameters = ntwrk.s
        frequencies_b.append(frequency_vector)
        parameters_b.append(s_parameters)

    # We have total of 38 different frequencies for each sample!
    # So, I am thinking to either flatten and taking the length of the vector, or take the sum of the s-parameters -then take magnitude of the obtained complex numbers!

    # Let me try to get the sum of the matrix!

    all_parameters_sum_b = []
    patient_each = {}
    all_paramters_sum_eigen = []
    for ind2, each in enumerate(parameters_b):
        out_sub = []
        out_eigen = []
        for every in each:
            each_ = list(np.array([abs(i.real) for i in list(every)]).reshape(-1))
            # Let me not try to calculate each eigen values and find the total eigen matrix, where I can take the average of them
            a, b = np.linalg.eig(np.array([i.real for i in list(every)]))
            a_sum = sum([abs(i) for i in list(a)])
            # print(a_sum)

            out_sub.append(each_)
            out_eigen.append(a_sum)
        all_parameters_sum_b.append(out_sub)
        all_paramters_sum_eigen.append(out_eigen)
    # print(all_parameters_sum_b)

    data_frame = pd.DataFrame(all_parameters_sum_b)
    data_eigen = pd.DataFrame(all_paramters_sum_eigen)
    # print(data_frame)
    # print(data_eigen)

    total = []
    for each in data_eigen.columns[:-17]:
        part_ = data_eigen[each].sum()
        total.append(part_)
    tot_ = sorted(total, reverse=True)
    to_ = [total.index(i) for i in tot_]
    print(to_)

    data_frame["Age"] = data_feature["Age"]
    data_frame["Pathology"] = data_feature["Pathology"]
    data_frame["BreastSize"] = data_feature["BreastSize"]
    data_frame["PatientName"] = data_feature["PatientName"]
    print(data_frame)

    # my_data_ = pd.DataFrame()
    # for each in data_frame.columns[:-4]:
    #
    #     # Let me remove the zeroes first!
    #
    #     mu_ = []
    #
    #     for every in data_frame[each].to_list():
    #         z_ = [i for i in every if i!=0.0]
    #         mu_.append(z_)
    #     data_frame[each] = mu_
    #
    #
    #
    #     # Let me first that the z-scores of each data! Then, I take some of outliers!
    #     # data_frame[each] = [list(stats.zscore(i)) for i in data_frame[each].to_list()]
    #
    #
    #
    #     # my_list_ = [sorted(i)[-100:] for i in data_frame[each].to_list()]
    #
    #     my_list_ = data_frame[each].to_list()
    #     print(my_list_)
    #     print(len(my_list_))
    #     data_frame[each] = my_list_
    # print(data_frame.columns)
    # my_data_ = data_frame
    # # my_data_["Age"] = data_frame["Age"]
    # # my_data_["Pathology"] = data_frame["Pathology"]
    # # my_data_["BreastSize"] = data_frame["BreastSize"]
    # # print(data_frame)
    # # print(my_data_)
    #
    # age_ = data_frame["Age"].to_list()
    # label_ = data_frame["Pathology"].to_list()
    # size_ = data_frame["BreastSize"].to_list()
    # name_ = data_frame["PatientName"].to_list()
    # #
    # # print(data_frame)
    # # # #
    # # # #
    # my_data_ = my_data_.explode(list(my_data_.columns[:-4]))
    # my_data_.reset_index(inplace=True)
    # data_frame = my_data_[my_data_.columns[1:]]
    # # # print(data_frame)
    # # data_frame = data_frame.loc[(data_frame != 0.0).all(axis=1)]
    # # data_frame.reset_index(inplace=True)
    # # data_frame = data_frame[data_frame.columns[1:]]
    # # # print(data_frame)

    return data_frame


data_frame = dataframe_create_with_s_parameters_sum(data_feature)

data_frame.sort_values(by="PatientName",inplace=True)
data_frame.to_excel("C:/Users/yurts/Documents/MITOS/overall_result.xlsx")
p_list = []
c_list = []
qr_list = []
for p_val in range(1,50,20):
    p_val = p_val/100
    for corr_v in range(1,50,20):
        corr_v = corr_v/100
        for qr_ in range(70,95,10):
            for i in range(1,2):

                def check_positions_for_each_patient(data_frame):

                    # Let me check each frequency, find the distribution of the difference!
                    ind_frequency_ = 0
                    unique_names = data_frame["PatientName"].unique().tolist()
                    data_all_h = pd.DataFrame()
                    data_all_nh = pd.DataFrame()
                    for each in data_frame.columns[:-4]:
                        each_name = []
                        each_column_healthy = []
                        each_column_nonhealthy = []
                        for every_name in unique_names:
                            print(data_frame)
                            print(every_name)
                            # print(each)
                            grouped_ = data_frame.sort_values(["Pathology"]).groupby("PatientName").get_group(every_name)[each].to_list() # .get_group(every_name)[each].to_list()
                            # print(grouped_)
                            ##Let me get the individuals, who have more than one breast!
                            if len(grouped_) == 2:
                                group1_ = grouped_[0]
                                group1_ = [i for i in group1_ if i!=0.0]
                                group2_ = grouped_[1]
                                group2_ = [i for i in group2_ if i != 0.0]

                                # Let me take the values which have smaller values in group1_
                                group1 = []
                                group2 = []
                                ind = 0
                                for each1 in group1_:
                                    if each1>group2_[ind]:
                                        group1.append(each1)
                                    ind = ind + 1
                                ind2 = 0
                                for each2 in group2_:
                                    if each2<group1_[ind2]:
                                        group2.append(each2)
                                    ind2 = ind2 + 1

                                differ_ = [a - b for a, b in zip(group1, group2)]
                                # print(differ_)
                                # differ_ = [i for i in differ_]
                                # Let me now, remove the zero elements!
                                difference_ = [abs(i) for i in differ_ if i!=0.0]
                                print("The number of difference is = {}".format(len(difference_)))


                                # Let me find the threshold value!

                                QR1= np.percentile(difference_,25)
                                QR3= np.percentile(difference_,qr_)
                                IQR = QR3-QR1
                                threshold = QR3
                                # print(threshold)

                                positions_ = []


                                my_ranked_list = sorted(difference_,reverse=True)
                                for evr in my_ranked_list:
                                    ind_ = difference_.index(evr)
                                    positions_.append(ind_)
                                    positions_ = positions_[:100]

                                sub_healthy = []
                                sub_nonhealthy = []
                                subname_ = []
                                for every_ in positions_:
                                    sub_healthy.append(group1[every_])
                                    sub_nonhealthy.append(group2[every_])
                                    subname_.append(every_name)

                                each_column_healthy.append(sub_healthy)
                                each_column_nonhealthy.append(sub_nonhealthy)
                                each_name.append(subname_)


                                # print(positions_)
                                # print(len(positions_))
                                # print(every_name)
                                # print(each)

                                # difference_ = list(stats.zscore(difference_))
                                # print(difference_)

                                # Let me plot the data!
                                # sns.kdeplot(difference_)
                                # plt.axvline(x=threshold,ymin=0,ymax=40)
                                # plt.ylim(0,42)
                                # plt.show()
                        data_all_h[each] = pd.Series(each_column_healthy)
                        data_all_nh[each] = pd.Series(each_column_nonhealthy)
                        # data_all_h["PatientName"] = pd.Series(each_name)
                        # data_all_nh["PatientName"] = pd.Series(each_name)



                        ind_frequency_ = ind_frequency_ + 1

                    # print(data_all_h["PatientName"])
                    # print(data_all_nh["PatientName"])

                    data_all_h.to_excel("C:/Users/yurts/Documents/MITOS/healthy_samples.xlsx")
                    data_all_nh.to_excel("C:/Users/yurts/Documents/MITOS/nonhealthy_samples.xlsx")

                    return data_all_h,data_all_nh
                data_all_h,data_all_nh = check_positions_for_each_patient(data_frame)

                # Let me now, see the created data frames and compare each of the frequencies to prepare out dataset for the modelling part!
                def my_data_preprocess(data_all_h,data_all_nh):

                    label_h = ["Healthy"]*len(data_all_h)
                    label_nh = ["Nonhealthy"]*len(data_all_nh)

                    data_all_h["Pathology"] = label_h
                    data_all_nh["Pathology"] = label_nh

                    data_concatanated_ = pd.concat([data_all_h,data_all_nh])
                    data_concatanated_.reset_index(inplace=True)
                    data_concatanated_ = data_concatanated_[data_concatanated_.columns[1:]]

                    # print(data_concatanated_)
                    # data_concatanated_.to_excel("C:/Users/yurts/Documents/MITOS/concatanated.xlsx")
                    # data_concatanated_ = data_concatanated_.explode(list(data_concatanated_.columns[:-1]))
                    # data_concatanated_.reset_index(inplace=True)
                    # data_concatanated_ = data_concatanated_[data_concatanated_.columns[1:]]

                    # Let me take the mean of each list in the frequencies and apply the model

                    for each in data_concatanated_.columns[:-1]:
                        val_ = data_concatanated_[each].to_list()
                        print(val_)
                        mean_val = [np.std(i) for i in val_]
                        data_concatanated_[each] = mean_val


                    print("data concatanated = {}".format(data_concatanated_))
                    data_concatanated_.to_excel("C:/Users/yurts/Documents/MITOS/concatanated_explode_std.xlsx")





                    return #data_significant, X_train, X_test, y_train, y_test,patient_number_test
                my_data_preprocess(data_all_h,data_all_nh) #data_parameter, X_train, X_test, y_train, y_test,patient_number_test

