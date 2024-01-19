import pandas as pd
import os
import skrf as rf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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
from  sklearn import ensemble
from  sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier


# Enter path of the most recent patient list!
path_patient_new = "C:/Users/ergLab/OneDrive/MITOS/MITOS/current_patient_status/New_patients.xlsx"

# Enter the path of the data input directory!
path_data_input = "C:/Users/ergLab/OneDrive/MITOS/MITOS/data/files/"

# Enter the path of the features!

path_feature = "C:/Users/ergLab/OneDrive/MITOS/MITOS/data_frame_features.xlsx"

def reader_(path_patient_new):

    # Let me read the data
    data_ = pd.read_excel(path_patient_new)

    # Let me take only the ones which have biopsy, also, the ones either malignant or benign
    data_ = data_[data_["BX_SCR"]=="bx"]
    data_ = data_[(data_["Pathology"]=="malignant")|(data_["Pathology"]=="benign")]
    data_.reset_index(inplace=True)
    data_=data_[data_.columns[1:]]
    print("Number of benign patient = {}".format(len(data_[data_["Pathology"]=="benign"])))
    print("Number of malignant patient = {}".format(len(data_[data_["Pathology"]=="malignant"])))

    return data_

data_ = reader_(path_patient_new)


def paths_taker(data_,path_data_input):

    # Let me get the s-parameters data!

    all_dir_ = os.listdir(path_data_input)
    all_data_ = []
    for each in all_dir_:
        sub_dirs = os.listdir(path_data_input + each)
        path_all = [path_data_input + each+"/"+i for i in sub_dirs]
        all_data_ = all_data_+path_all
    check_data = [i.lower() for i in all_data_]
    data_["PatientName"]= [i.lower() for i in data_["PatientName"]]

    my_directories = []
    counter=0
    for each in check_data:
        for every in data_["PatientName"].to_list():
            if every in each:
                my_directories.append(all_data_[counter])
        counter=counter+1
    print(len(my_directories))

    my_paths = []
    for each in my_directories:
        if os.listdir(each)[0]=="temp.s2p" and len(os.listdir(each))==1:
            continue
        elif os.listdir(each)[0]=="temp.s2p" and len(os.listdir(each))>1:
            my_paths.append(each + "/" + os.listdir(each)[1])
        else:
            my_paths.append(each+ "/"+os.listdir(each)[0])

    print(len(my_paths))

    return my_paths
my_paths = paths_taker(data_,path_data_input)

print("Number of healthy samples = {}".format(len(my_paths)-len(data_)))



def path_to_file(my_paths,path_feature):
    # Let me put the data into data frame and extract both names_surnames as well as the left and right sides!
    name_second_name = ['_'.join(os.path.split(i)[1].split("_")[0:2]).lower() for i in my_paths]
    breast_side = [os.path.split(i)[1].split("_")[2].lower()[0] for i in my_paths]

    data_path_ = pd.DataFrame({"PatientName": name_second_name, "Side": breast_side, "paths": my_paths})
    data_feature = pd.read_excel(path_feature)

    data_feature = data_feature.sort_values(by=["PatientName","Side"])
    data_feature.reset_index(inplace=True)
    data_feature = data_feature[data_feature.columns[1:]]

    data_path_ = data_path_.sort_values(by=["PatientName","Side"])
    data_path_.reset_index(inplace=True)
    data_path_ = data_path_[data_path_.columns[1:]]

    data_feature["paths"] = data_path_["paths"]

    return data_feature

data_feature = path_to_file(my_paths,path_feature)


# Now, let me get the parameters!
#

def dataframe_create_with_s_parameters(data_feature):

    my_paths = data_feature["paths"].to_list()
    print(my_paths)

    frequencies = []
    parameters = []
    for each in my_paths:
        ntwrk = rf.Network(each)
        frequency_vector = ntwrk.frequency.f
        s_parameters = ntwrk.s
        frequencies.append(frequency_vector)
        parameters.append(s_parameters)



    print(len(parameters[0]))
    print(len(frequencies[0]))


    # We have total of 38 different frequencies for each sample!
    # So, I am thinking to either flatten and taking the length of the vector, or take the sum of the s-parameters -then take magnitude of the obtained complex numbers!

    # Let me try to get the sum of the matrix!

    all_parameters_sum=[]
    patient_each = []
    for each in parameters:
        out_sub = []
        for every in each:
            sub = []
            for s in every:
                dot = np.vdot(s[0],s[1])
                for ind,j in enumerate(s):
                    if ind>=2:
                        if s[ind]!=0:
                            dot = np.vdot(s[ind],dot)

                sub.append(abs(dot))

            print(sub)
            out_sub.append(sum(sub))
        patient_each.append(out_sub)

    print(len(patient_each))

    # Let me now get the frequencies of the matrix!

    frequencies_each = []
    for freq in frequencies[0]:
        frequencies_each.append(freq)

    data_frame_s_parameters = pd.DataFrame(patient_each)


    print(list(data_frame_s_parameters.columns))
    data_frame_s_parameters.columns = ["Frequency "+str(i) for i in list(data_frame_s_parameters.columns)]
    print(data_frame_s_parameters)

    return data_frame_s_parameters

data_frame_s_parameters = dataframe_create_with_s_parameters(data_feature)







# # Now, let me create the concluded excel_file, with the including data!
#
def create_table(data_feature,data_frame_s_parameters):

    # data_feature.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/features_.xlsx")
    # data_frame_s_parameters.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/s_parameters_.xlsx")

    data_frame_s_parameters["PatientName"] = data_feature["PatientName"]
    data_frame_s_parameters["Age"] = data_feature["Age"]
    data_frame_s_parameters["Size"] = data_feature["BreastSize_x"]
    data_frame_s_parameters["Side"] = data_feature["Side"]
    data_frame_s_parameters["Density"] = data_feature["Density"]
    data_frame_s_parameters["Pathology"] = data_feature["Pathology"]

    return

create_table(data_feature,data_frame_s_parameters)


print(data_frame_s_parameters)


#
# def preprocess_(data_frame_s_parameters):
#
#     additional_info_ = {}
#
#     # Let me do the encoding, encode the categorical variables to the numeric ones, for side and pathology for now!
#     label_encoder = LabelEncoder()
#     # left->0, right->1
#     data_frame_s_parameters["Side"]= label_encoder.fit_transform(data_frame_s_parameters["Side"])
#
#
#     #  Let me get only malignant and benign ones for now!
#
# #     data_frame_s_parameters = data_frame_s_parameters[(data_frame_s_parameters["Pathology"]=="benign")|(data_frame_s_parameters["Pathology"]=="malignant")]
# #     data_frame_s_parameters.reset_index(inplace=True)
# #     data_frame_s_parameters = data_frame_s_parameters[data_frame_s_parameters.columns[1:]]
# #
# #     # 0-> benign, 1-> malignant
# #     data_frame_s_parameters["Pathology"]= label_encoder.fit_transform(data_frame_s_parameters["Pathology"])
# #     # Let me now, encode 1-2 to 0 and 3-4 to 1, for the size!
# #     data_frame_s_parameters.loc[(data_frame_s_parameters["Size"]==1)|(data_frame_s_parameters["Size"]==2),"Size_new"]=int(0)
# #     data_frame_s_parameters.loc[(data_frame_s_parameters["Size"]==3)|(data_frame_s_parameters["Size"]==4),"Size_new"]=int(1)
# #     data_frame_s_parameters["Size_new"] = data_frame_s_parameters["Size_new"].astype(int)
# #
# #
# #     additional_info_["Density"] = data_frame_s_parameters["Density"]
# #     additional_info_["Size"] = data_frame_s_parameters["Size"]
# #     additional_info_["Age"] = data_frame_s_parameters["Age"]
# #     #
# #     # Let me take out the size and density!
# #     del  data_frame_s_parameters["Size"]
# #     del  data_frame_s_parameters["Density"]
# #     del data_frame_s_parameters["PatientName"]
# #
#     return data_frame_s_parameters, additional_info_
#
# data_frame_s_parameters, additional_info_ = preprocess_(data_frame_s_parameters)
#
#






# def feature_selection_distribution_visualization(data_frame_s_parameters):
#
#     # data_frame_s_parameters.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/feature_vec.xlsx")
#     # Let me look at the distributions of the s-parameters!
#     #
#     # for each in range(0,38):
# #         data_frame_s_parameters["Frequency {}".format(each)] = np.log2(data_frame_s_parameters["Frequency {}".format(each)])
# #         sns.distplot(data_frame_s_parameters["Frequency {}".format(each)])
# #         plt.show()
# #     #
# #     #
# #     # 0-> Healthy, 1-> Benign, 2-> Malignant
# #     healthy = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==0]
# #     benign = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==1]
# #     malignant = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==2]
# #     #
# # #    Let me check the age distributions!
# #     #
# #     bins = np.linspace(0, 200, 40)
# #
# #     plt.hist(healthy["Age"], bins, label="healthy")
# #     plt.hist(benign["Age"], bins, label="benign")
# #     plt.hist(malignant["Age"], bins, label="malignant")
# #     plt.xlabel("Ages",fontsize=18)
# #     plt.ylabel("Frequency",fontsize=18)
# #     plt.xticks(fontsize=18)
# #     plt.yticks(fontsize=18)
# #     plt.legend(fontsize=16)
# #     plt.show()
# #     #
# # #    Let me check breast size distribution!
# #     #
# #     size_healthy_s = len(healthy[healthy["Size_new"]==0])
# #     size_healthy_b =len(healthy)-len(healthy[healthy["Size_new"]==0])
# #     size_benign_s = len(benign[benign["Size_new"] == 0])
# #     size_benign_b = len(benign) - len(benign[benign["Size_new"] == 0])
# #     size_malignant_s = len(malignant[malignant["Size_new"] == 0])
# #     size_malignant_b = len(malignant) - len(malignant[malignant["Size_new"] == 0])
# #     #
# #     # create data
# #     x = ['Healthy', 'Benign', 'Malignant']
# #     small = [size_healthy_s,size_benign_s,size_malignant_s]
# #     big = [size_healthy_b,size_benign_b,size_malignant_b]
# #
# #
# #     # plot bars in stack manner
# #     plt.bar(x, small, color='orange',label="Small Breast")
# #     plt.bar(x, big, bottom=small, color='purple',label="Big Breast")
# #
# #     plt.ylabel("Number of Samples",fontsize=18)
# #     plt.xticks(fontsize=18)
# #     plt.yticks(fontsize=18)
# #     plt.legend(fontsize=16)
# #
# #     plt.show()
# #
# #
# #
# #     # Let me check the Breast Side Distribution
# #     side_healthy_l = len(healthy[healthy["Side"]==0])
# #     side_healthy_r =len(healthy)-len(healthy[healthy["Side"]==0])
# #     side_benign_l = len(benign[benign["Side"] == 0])
# #     side_benign_r = len(benign) - len(benign[benign["Side"] == 0])
# #     side_malignant_l = len(malignant[malignant["Side"] == 0])
# #     side_malignant_r = len(malignant) - len(malignant[malignant["Side"] == 0])
# #
# #
# #
# #     # create data for side distribution
# #     x = ['Healthy', 'Benign', 'Malignant']
# #     left = [side_healthy_l,side_benign_l,side_malignant_l]
# #     right = [side_healthy_r,side_benign_r,side_malignant_r]
# #
# #
# #     # plot bars in stack manner
# #     plt.bar(x, left, color='orange',label="Left Breast")
# #     plt.bar(x, right, bottom=left, color='purple',label="Right Breast")
# #     plt.ylabel("Number of Samples",fontsize=18)
# #     plt.xticks(fontsize=18)
# #     plt.yticks(fontsize=18)
# #     plt.legend(fontsize=16)
# #
# #     plt.show()
# #
#     return
# # feature_selection_distribution_visualization(data_frame_s_parameters)
# #
# #
# def  check_normality_in_continuous_variables(data_frame_s_parameters):
# #    Let me check the normality with shapiro wilk test!
#     for each in range(0,38):
#         freq0_n = stats.shapiro(data_frame_s_parameters["Frequency {}".format(each)])[1]
#         if freq0_n>=0.05:
#             data_frame_s_parameters["Frequency {}".format(each)] = np.log10(data_frame_s_parameters["Frequency {}".format(each)])
#             if stats.shapiro(data_frame_s_parameters["Frequency {}".format(each)])[1]>=0.05:
#                 print("S-parameters for frequency {} is not normally distributed!".format(each))
#
#     # Let me check the normality in ages!
#     if stats.shapiro(data_frame_s_parameters["Age"])[1]>=0.05:
#         data_frame_s_parameters["Age"] = np.log10(data_frame_s_parameters["Age"])
#
#     data_corr = data_frame_s_parameters.corr(method="pearson")
#
#     # for each in data_frame_s_parameters.columns:
#     #     corr_
#
#     return data_frame_s_parameters
#
# data_frame_s_parameters = check_normality_in_continuous_variables(data_frame_s_parameters)
#
#
# def check_correlation_class_label(data_frame_s_parameters):
#
#     freqs_to_remove = []
#     coefs_pval =[]
#
#     for each in range(0,38):
#
#         coeff = stats.pointbiserialr(data_frame_s_parameters["Frequency {}".format(each)],
#                                  data_frame_s_parameters["Pathology"])[1]
#
#         coefs_pval.append(coeff)
#
#         # Let me filter the values by correlation, when we take the correlation p-values<0.05
#         if coeff>=0.05:
#             freqs_to_remove.append("Frequency {}".format(each))
#
#     # Now, let me plot the point biserial correlations of each frequency with the pathology of the sample!
#     fs = []
#     for each in range(0,38):
#         fs.append("f{}".format(each))
#
#
#     plt.scatter(fs,coefs_pval,marker=".")
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel("Frequencies",fontsize=18)
#     plt.ylabel("Point Biserial Correlation p-values",fontsize=18)
#     plt.show()
#
#     count=0
#     my_freq_dict_pval ={}
#     for each in range(0,38):
#         my_freq_dict_pval["Frequency {}".format(each)]=coefs_pval[count]
#
#         count=count+1
#
#     data_frame_s_parameters.drop(freqs_to_remove, axis=1, inplace=True)
#
#     return freqs_to_remove,my_freq_dict_pval
#
# freqs_to_remove,my_freq_dict_pval = check_correlation_class_label(data_frame_s_parameters)
#
#
#
# def check_correlation_between_attributes_(data_frame_s_parameters,my_freq_dict_pval):
#
#     # Pearson correlation or LDA between attributes!
#
#     # Let me try the pearson correlation!
#
#     corr = data_frame_s_parameters.corr(method="pearson")
#     correlate=corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
#     sns.heatmap(correlate,cmap="YlGnBu",annot=True)
#     plt.show()
#
#     # Let me put the coeff cutoff of 0.7, and find the attributes, which are similar to each other
#
#     list_correlation_remove = []
#     for each in range(0,len(corr)):
#         counter=0
#         for every in corr[corr.columns[each]]:
#             sub=[]
#             if every>0.7:
#                 sub.append([corr.columns[each]][0])
#                 sub.append(corr.index[counter])
#                 list_correlation_remove.append(sub)
#             counter=counter+1
#
#     count = 0
#     for every in list_correlation_remove:
#         if every[0]==every[1]:
#             list_correlation_remove.pop(count)
#         count=count+1
#
#
#
#     # Let me remove one of the columns that are similar to each other and have lower p value, when doing the correlation against the labels!
#
#
#     for each in list_correlation_remove:
#         if "Age" not in each:
#             if "Pathology" not in each:
#                 if "Side" not in each:
#                     if "Size_new" not in each:
#                         if my_freq_dict_pval[each[0]]>my_freq_dict_pval[each[1]]:
#                             if each[0] in data_frame_s_parameters.columns:
#                                 data_frame_s_parameters.drop([each[0]],axis=1,inplace=True)
#                         if my_freq_dict_pval[each[0]] < my_freq_dict_pval[each[1]]:
#                             if each[1] in data_frame_s_parameters.columns:
#                                 data_frame_s_parameters.drop([each[1]], axis=1,inplace=True)
#
#     print("Total number of the attributes is {}".format(len(data_frame_s_parameters.columns)-1))
#
#     return data_frame_s_parameters
#
# data_frame_s_parameters = check_correlation_between_attributes_(data_frame_s_parameters,my_freq_dict_pval)
# print(data_frame_s_parameters)
#
# # Now, let me do the scaling, we need to scale our continuous variables to have them in a certain scales!
#
# def scaler(data_frame_s_parameters):
#
#     print(data_frame_s_parameters[data_frame_s_parameters.columns[:-3]])
#     # Let me scale the s-parameters and age!
#
#     scale = StandardScaler()
#     data_frame_s_parameters[data_frame_s_parameters.columns[:-3]] = scale.fit_transform(data_frame_s_parameters[data_frame_s_parameters.columns[:-3]])
#
#     # data_frame_s_parameters = shuffle(data_frame_s_parameters)
#     # data_frame_s_parameters.reset_index(inplace=True)
#     # data_frame_s_parameters = data_frame_s_parameters[data_frame_s_parameters.columns[1:]]
#
#
#
#     # Let me get the attributes from the parameters!
#
#     Attribute_ = data_frame_s_parameters[data_frame_s_parameters.columns[:-2]]
#     print(data_frame_s_parameters.columns)
#
#     Attribute_["Size"] = data_frame_s_parameters["Size_new"]
#     print(Attribute_.columns)
#     print(Attribute_)
#     Attribute_ = np.array(Attribute_)
#
#     Label_ = data_frame_s_parameters["Pathology"]
#     Label_ = np.array(Label_)
#
#
#
#     return Attribute_,Label_
#
# Attribute_,Label_ = scaler(data_frame_s_parameters)
#
#
# # def decision_tree_model(Attribute_,Label_):
# #
# #     # create a dictionary of all values we want to test
# #     param_grid = {'criterion': ['gini', 'entropy','log_loss'],'splitter':['best', 'random'], 'max_depth': np.arange(3, 15),
# #                   'random_state':np.arange(0,43)}
# #
# #     # decision tree model
# #     dtree_model = DecisionTreeClassifier()
# #
# #     #use gridsearch to test all values
# #     dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=5)
# #     #fit model to data
# #     dtree_gscv.fit(Attribute_, Label_)
# #     parameter__ = dtree_gscv.best_params_
# #     print(parameter__)
# #
# #
# #     clf = DecisionTreeClassifier(criterion=parameter__["criterion"],max_depth=parameter__["max_depth"],
# #                                             splitter=parameter__["splitter"],random_state=parameter__["random_state"])
# #
# #     # clf = DecisionTreeClassifier(random_state=42)
# #     k_folds = KFold(n_splits=5,shuffle=True)
# #
# #     scores_tree = cross_val_score(clf, Attribute_, Label_, cv=k_folds)
# #
# #     print("Decision Tree Cross Validation Scores: ", scores_tree)
# #     print("Decision Tree Average CV Score: ", scores_tree.mean())
# #     print("Decision Tree Number of CV Scores used in Average: ", len(scores_tree))
# #
# #     # Let me draw the confusion matrix!
# #     y_predict = cross_val_predict(clf,Attribute_,Label_,cv=5)
# #     conf_matrix = confusion_matrix(Label_, y_predict)
# #
# #     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
# #     plt.xlabel("Predicted Label",fontsize=18)
# #     plt.ylabel("True Label",fontsize=18)
# #     plt.xticks(fontsize=18)
# #     plt.yticks(fontsize=18)
# #     plt.title("Decision Tree Confusion Matrix",fontsize=18)
# #
# #     plt.show()
# #
# #     return scores_tree
# #
# # scores_tree = decision_tree_model(Attribute_,Label_)
#
#
# # def adaboost_decision_tree_model(Attribute_,Label_):
# #
# #     # create a dictionary of all values we want to test
# #
# #     dt = DecisionTreeClassifier()
# #     rf = ensemble.RandomForestClassifier()
# #     svc = svm.SVC()
# #
# #
# #
# #     param_grid = {'n_estimators': np.arange(1,60,2),'learning_rate':np.arange(0.1,1.6,0.05),
# #                   'algorithm':['SAMME','SAMME.R'],
# #                   'base_estimator':[dt,rf,svc]}
# #
# #     # decision tree model
# #     abc =AdaBoostClassifier()
# #
# #     #use gridsearch to test all values
# #     dtree_gscv = GridSearchCV(abc, param_grid, cv=4,scoring='neg_mean_absolute_error',n_jobs=-1,
# #                               return_train_score=True)
# #
# #     #fit model to data
# #     dtree_gscv.fit(Attribute_, Label_)
# #     parameter__ = dtree_gscv.best_params_
# #     print(parameter__)
# #
# #
# #     clf = AdaBoostClassifier(n_estimators=parameter__["n_estimators"],learning_rate=parameter__["learning_rate"],algorithm=parameter__["algorithm"],
# #                              base_estimator=parameter__["base_estimator"])
# #
# #     # clf = DecisionTreeClassifier(random_state=42)
# #     k_folds = KFold(n_splits=4,shuffle=True)
# #
# #     scores_tree = cross_val_score(clf, Attribute_, Label_, cv=k_folds)
# #
# #     print("Decision Tree Cross Validation Scores: ", scores_tree)
# #     print("Decision Tree Average CV Score: ", scores_tree.mean())
# #     print("Decision Tree Number of CV Scores used in Average: ", len(scores_tree))
# #
# #     # Let me draw the confusion matrix!
# #     y_predict = cross_val_predict(clf,Attribute_,Label_,cv=4)
# #     conf_matrix = confusion_matrix(Label_, y_predict)
# #
# #     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
# #     plt.xlabel("Predicted Label",fontsize=18)
# #     plt.ylabel("True Label",fontsize=18)
# #     plt.xticks(fontsize=18)
# #     plt.yticks(fontsize=18)
# #     plt.title("Decision Tree Confusion Matrix",fontsize=18)
# #
# #     plt.show()
# #
# #     return
#
# adaboost_decision_tree_model(Attribute_,Label_)


#
# def gboosting_classification_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'loss': ['auto', 'binary_crossentropy','log_loss','categorical_crossentropy'],'learning_rate': np.arange(0.1, 0.8),
#                   'random_state':np.arange(0,43)}
#
#     # Gradient Boosting model
#     gradient_tree_model = GradientBoostingClassifier()
#
#     #use gridsearch to test all values
#     gradient_gscv = GridSearchCV(gradient_tree_model, param_grid, cv=5)
#     #fit model to data
#     gradient_gscv.fit(Attribute_, Label_)
#     parameter__ = gradient_gscv.best_params_
#     print(parameter__)
#
#
#     clf_gradient = GradientBoostingClassifier(loss=parameter__["loss"],
#                                             learning_rate=parameter__['learning_rate'],random_state=parameter__["random_state"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_grd = cross_val_score(clf_gradient, Attribute_, Label_, cv=k_folds)
#
#     print("Gradient Boosting Cross Validation Scores: ", scores_grd)
#     print("Gradient Boosting Decision Tree Average CV Score: ", scores_grd.mean())
#     print("Gradient Boosting Decision Tree Number of CV Scores used in Average: ", len(scores_grd))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_gradient,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Gradient Boosting Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_grd
#
# scores_grd = gboosting_classification_model(Attribute_,Label_)
#
# # #
# # #
# # #
# # #


# def hboosting_classification_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'loss': ['auto', 'binary_crossentropy','log_loss','categorical_crossentropy'],'learning_rate': np.arange(0.1, 0.8),
#                   'random_state':np.arange(0,43)}
#
#     # Gradient Boosting model
#     hgradient_tree_model = HistGradientBoostingClassifier()
#
#     #use gridsearch to test all values
#     hgradient_gscv = GridSearchCV(hgradient_tree_model, param_grid, cv=5)
#     #fit model to data
#     hgradient_gscv.fit(Attribute_, Label_)
#     parameter__ = hgradient_gscv.best_params_
#     print(parameter__)
#
#
#     clf_hgradient = HistGradientBoostingClassifier(loss=parameter__["loss"],
#                                             learning_rate=parameter__['learning_rate'],random_state=parameter__["random_state"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_hgrd = cross_val_score(clf_hgradient, Attribute_, Label_, cv=k_folds)
#
#     print("Hist Gradient Boosting Cross Validation Scores: ", scores_hgrd)
#     print("Hist Gradient Boosting Decision Tree Average CV Score: ", scores_hgrd.mean())
#     print("Hist Gradient Boosting Decision Tree Number of CV Scores used in Average: ", len(scores_hgrd))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_hgradient,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Hist Gradient Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_hgrd
#
# scores_hgrd = hboosting_classification_model(Attribute_,Label_)


# # # # Let me try SVC

# def svm_classification_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'kernel': ['linear', 'poly','rbf','sigmoid'],'degree': np.arange(2, 6),'gamma': ['scale','auto'],'decision_function_shape': ['ovo','ovr'],
#                   'random_state':np.arange(0,43)}
#
#     # Gradient Boosting model
#     svm_svc_model = svm.SVC()
#
#     #use gridsearch to test all values
#     svm_svc_gscv = GridSearchCV(svm_svc_model, param_grid, cv=3)
#     #fit model to data
#     svm_svc_gscv.fit(Attribute_, Label_)
#     parameter__ = svm_svc_gscv.best_params_
#     print(parameter__)
#
#
#     clf_svm_svc = svm.SVC(kernel=parameter__["kernel"], gamma=parameter__['gamma'],decision_function_shape=parameter__["decision_function_shape"],
#                                             degree=parameter__['degree'],random_state=parameter__["random_state"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=3,shuffle=True)
#
#     scores_svc = cross_val_score(clf_svm_svc, Attribute_, Label_, cv=k_folds)
#
#     print("SVC Cross Validation Scores: ", scores_svc)
#     print("SVC Average CV Score: ", scores_svc.mean())
#     print("SVC Number of CV Scores used in Average: ", len(scores_svc))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_svm_svc,Attribute_,Label_,cv=3)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("SVM Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_svc
#
# scores_svc = svm_classification_model(Attribute_,Label_)

# # #
# # #
# # #
# # #
# # #
# # # # Let me try Linear SVC
# # #
# #
# def linear_svc_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'penalty': ['l1', 'l2'],'max_iter': np.arange(500,1001,250),'loss': ['hinge','squared_hinge'],
#                   'random_state':np.arange(0,43)}
#
#     # Gradient Boosting model
#     svm_linearsvc_model = svm.LinearSVC()
#
#     #use gridsearch to test all values
#     svm_linearsvc_gscv = GridSearchCV(svm_linearsvc_model, param_grid, cv=5)
#     #fit model to data
#     svm_linearsvc_gscv.fit(Attribute_, Label_)
#     parameter__ = svm_linearsvc_gscv.best_params_
#     print(parameter__)
#
#
#     clf_linearsvm_svc = svm.LinearSVC(penalty=parameter__["penalty"], max_iter=parameter__['max_iter'],loss=parameter__["loss"],
#                                             random_state=parameter__["random_state"])
#
#
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_lsvc = cross_val_score(clf_linearsvm_svc, Attribute_, Label_, cv=k_folds)
#
#     print("Linear SVC Cross Validation Scores: ", scores_lsvc)
#     print("Linear SVC Average CV Score: ", scores_lsvc.mean())
#     print("Linear SVC Number of CV Scores used in Average: ", len(scores_lsvc))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_linearsvm_svc,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Linear SVM Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_lsvc
#
# scores_lsvc = linear_svc_model(Attribute_,Label_)
#
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# #
# # # Let me try Nu-support vector classifier
#
# def nu_classification_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'kernel': ['linear', 'poly','rbf','sigmoid'],'degree': np.arange(2, 6),'gamma': ['scale','auto'],'decision_function_shape': ['ovo','ovr'],
#                   'random_state':np.arange(0,43)}
#
#     # Gradient Boosting model
#     nu_svc_model= svm.NuSVC()
#
#     #use gridsearch to test all values
#     nu_svc_gscv = GridSearchCV(nu_svc_model, param_grid, cv=5)
#     #fit model to data
#     nu_svc_gscv.fit(Attribute_, Label_)
#     parameter__ = nu_svc_gscv.best_params_
#     print(parameter__)
#
#
#     clf_nu_svc = svm.NuSVC(kernel=parameter__["kernel"], gamma=parameter__['gamma'],decision_function_shape=parameter__["decision_function_shape"],
#                                             degree=parameter__['degree'],random_state=parameter__["random_state"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_nsvc = cross_val_score(clf_nu_svc, Attribute_, Label_, cv=k_folds)
#
#     print("Nu-SVC Cross Validation Scores: ", scores_nsvc)
#     print("Nu-SVC Average CV Score: ", scores_nsvc.mean())
#     print("Nu-SVC Number of CV Scores used in Average: ", len(scores_nsvc))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_nu_svc,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Nu-SVM Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_nsvc
#
# scores_nsvc = nu_classification_model(Attribute_,Label_)
#
#

## Let me now, try neural network classifications!

# def mlp_classification_model(Attribute_,Label_,additional_info):
#
#     # # create a dictionary of all values we want to test
#     # param_grid = {'activation': ['identity','logistic','tanh','relu'],'solver':['lbfgs','sgd','adam','relu'],
#     #               'learning_rate':['constant','invscaling','adaptive'],
#     #               'random_state':np.arange(0,43),
#     #               'max_iter':np.arange(100,501,50),"hidden_layer_sizes":[ (i,) for i in list(np.arange(50,250,25))]}
#     #
#     # # Gradient Boosting model
#     # mlp_model= neural_network.MLPClassifier()
#     #
#     # #use gridsearch to test all values
#     # mlp_gscv = GridSearchCV(mlp_model, param_grid, cv=3)
#     # #fit model to data
#     # mlp_gscv.fit(Attribute_, Label_)
#     # parameter__ = mlp_gscv.best_params_
#     # print(parameter__)
#
#
#     # clf_mlp = neural_network.MLPClassifier(activation=parameter__["activation"],
#     #                                        solver=parameter__["solver"],
#     #                                        learning_rate=parameter__["learning_rate"],
#     #                                        random_state=parameter__["random_state"],max_iter=parameter__["max_iter"],hidden_layer_sizes=parameter__["hidden_layer_sizes"])
#
#     clf_mlp = neural_network.MLPClassifier(activation='relu',
#                                            solver='sgd',
#                                            learning_rate='constant',
#                                            random_state=36,max_iter=100,hidden_layer_sizes=(50,))
#
#
#     k_folds = KFold(n_splits=3,shuffle=True)
#
#     scores_mlp = cross_val_score(clf_mlp, Attribute_, Label_, cv=k_folds)
#
#     print("MLP Cross Validation Scores: ", scores_mlp)
#     print("MLP Average CV Score: ", scores_mlp.mean())
#     print("MLP Number of CV Scores used in Average: ", len(scores_mlp))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf_mlp,Attribute_,Label_,cv=3)
#     probability = cross_val_predict(clf_mlp,Attribute_,Label_,cv=3,method="predict_proba")
#     print(probability)
#
#     data_att = pd.DataFrame(Attribute_)
#     data_att["Size"] = additional_info["Size"]
#     data_att["Age"] = additional_info["Age"]
#     data_att["Density"] = additional_info["Density"]
#     data_att["Label"] = Label_
#     print(data_att)
#     data_att["Probabilities of Malignancy"] = list_probabas_ = [i[1] for i in probability]
#     data_att["Predict"] =y_predict
#
#     print(data_att)
#     #
#
#     conf_matrix = confusion_matrix(Label_, y_predict)
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu",cbar=False)
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("MLP Confusion Matrix All Breasts",fontsize=18)
#     plt.show()
#
#     return #scores_mlp
#
# mlp_classification_model(Attribute_,Label_,additional_info_)
#
#
#







# # # Let me apply kNN classification!
#
#
# def kNN_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'n_neighbors': np.arange(2,10,1),'weights':['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree','brute']}
#
#     # decision tree model
#     knn_model = neighbors.KNeighborsClassifier()
#
#     #use gridsearch to test all values
#     knn_gscv = GridSearchCV(knn_model, param_grid, cv=5)
#     #fit model to data
#     knn_gscv.fit(Attribute_, Label_)
#     parameter__ = knn_gscv.best_params_
#     print(parameter__)
#
#
#     clf = neighbors.KNeighborsClassifier(n_neighbors=parameter__["n_neighbors"],weights=parameter__["weights"],
#                                             algorithm=parameter__["algorithm"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_knn = cross_val_score(clf, Attribute_, Label_, cv=k_folds)
#
#     print("kNN Cross Validation Scores: ", scores_knn)
#     print("kNN Average CV Score: ", scores_knn.mean())
#     print("kNN Number of CV Scores used in Average: ", len(scores_knn))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("kNN Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_knn
#
# scores_knn = kNN_model(Attribute_,Label_)
#
#
#
#
#
#
# def logistic_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'solver': ['newton-cg', 'lbfgs','liblinear','sag','saga'] ,'penalty':['l1', 'l2','elasticnet','none']}
#
#     # decision tree model
#     logistic_model = linear_model.LogisticRegression()
#
#     #use gridsearch to test all values
#     logistic_gscv = GridSearchCV(logistic_model, param_grid, cv=5)
#     #fit model to data
#     logistic_gscv.fit(Attribute_, Label_)
#     parameter__ = logistic_gscv.best_params_
#     print(parameter__)
#
#
#     clf = linear_model.LogisticRegression(solver=parameter__["solver"],penalty=parameter__["penalty"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_lreg = cross_val_score(clf, Attribute_, Label_, cv=k_folds)
#
#     print("Logistic Cross Validation Scores: ", scores_lreg)
#     print("Logistic Average CV Score: ", scores_lreg.mean())
#     print("Logistic Number of CV Scores used in Average: ", len(scores_lreg))
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Logistic Regression Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_lreg
#
# scores_lreg = logistic_model(Attribute_,Label_)
#
#
#
#
# Let me do the random forest classification as well!

# def randomforest_model(Attribute_,Label_):
#
#     # create a dictionary of all values we want to test
#     param_grid = {'n_estimators': np.arange(50,151,25) ,'criterion':['gini', 'entropy','log_loss'],
#                   'max_features':['sqrt', 'log2','None']}
#
#     # decision tree model
#     rf_model = ensemble.RandomForestClassifier()
#
#     #use gridsearch to test all values
#     rf_gscv = GridSearchCV(rf_model, param_grid, cv=5)
#     #fit model to data
#     rf_gscv.fit(Attribute_, Label_)
#     parameter__ = rf_gscv.best_params_
#     print(parameter__)
#
#
#     clf = ensemble.RandomForestClassifier(n_estimators=parameter__["n_estimators"],criterion=parameter__["criterion"],
#                                           max_features=parameter__["max_features"])
#
#     # clf = DecisionTreeClassifier(random_state=42)
#     k_folds = KFold(n_splits=5,shuffle=True)
#
#     scores_rf = cross_val_score(clf, Attribute_, Label_, cv=k_folds)
#
#     print("Random Forest Cross Validation Scores: ", scores_rf)
#     print("Random Forest Average CV Score: ", scores_rf.mean())
#     print("Random Forest Number of CV Scores used in Average: ", len(scores_rf))
#
#
#     # Let me draw the confusion matrix!
#     y_predict = cross_val_predict(clf,Attribute_,Label_,cv=5)
#     conf_matrix = confusion_matrix(Label_, y_predict)
#
#     sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu")
#     plt.xlabel("Predicted Label",fontsize=18)
#     plt.ylabel("True Label",fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.title("Random Forest Confusion Matrix",fontsize=18)
#
#     plt.show()
#
#     return scores_rf
#
# scores_rf = randomforest_model(Attribute_,Label_)

#
#
# # Let me plot the cv scores im the column graph!
#
#
#
# scores_ = list(scores_tree) + list(scores_grd) + list(scores_hgrd) + list(scores_svc) + list(scores_lsvc) + list(scores_nsvc) + list(scores_mlp) + list(scores_knn) + \
#           list(scores_lreg) + list(scores_rf)
#
#
# def compare_models(scores_):
#     k = 5
#     model_names = ["DT"] * k + ["GrB"] * k + ["HB"] * k + ["SCV"] * k + ["LSCV"] * k + ["NSCV"] * k + ["MLP"] * k + [
#         "kNN"] * k + ["LR"] * k + ["RF"] * k
#
#     print(model_names)
#     print(len(model_names))
#     print(scores_)
#     print(len(scores_))
#
#     data_frame = pd.DataFrame({"Names": model_names, "Scores": scores_})
#     sns.boxplot(x=data_frame["Names"], y=data_frame["Scores"])
#
#     plt.xlabel("Model Names", fontsize=18)
#     plt.ylabel("Cross Validation Scores (5-fold)", fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#
#     plt.show()
#
#     return
#
# compare_models(scores_)
#
