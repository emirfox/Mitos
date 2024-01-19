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

                sub.append(abs(s.sum()))
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


# Now, let me create the concluded excel_file, with the including data!

def create_table(data_feature,data_frame_s_parameters):

    data_feature.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/features_.xlsx")
    data_frame_s_parameters.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/s_parameters_.xlsx")

    data_frame_s_parameters["PatientName"] = data_feature["PatientName"]
    data_frame_s_parameters["Age"] = data_feature["Age"]
    data_frame_s_parameters["Size"] = data_feature["BreastSize_x"]
    data_frame_s_parameters["Side"] = data_feature["Side"]
    data_frame_s_parameters["Density"] = data_feature["Density"]
    data_frame_s_parameters["Pathology"] = data_feature["Pathology"]

    return

create_table(data_feature,data_frame_s_parameters)



def preprocess_(data_frame_s_parameters):

    additional_info_ = {}

    # Let me do the encoding, encode the categorical variables to the numeric ones, for side and pathology for now!
    label_encoder = LabelEncoder()
    # left->0, right->1
    data_frame_s_parameters["Side"]= label_encoder.fit_transform(data_frame_s_parameters["Side"])


    #  Let me get only malignant and benign ones for now!

    data_frame_s_parameters = data_frame_s_parameters[(data_frame_s_parameters["Pathology"]=="benign")|(data_frame_s_parameters["Pathology"]=="malignant")]
    data_frame_s_parameters.reset_index(inplace=True)
    data_frame_s_parameters = data_frame_s_parameters[data_frame_s_parameters.columns[1:]]

    # 0-> benign, 1-> malignant
    data_frame_s_parameters["Pathology"]= label_encoder.fit_transform(data_frame_s_parameters["Pathology"])
    # Let me now, encode 1-2 to 0 and 3-4 to 1, for the size!
    data_frame_s_parameters.loc[(data_frame_s_parameters["Size"]==1)|(data_frame_s_parameters["Size"]==2),"Size_new"]=int(0)
    data_frame_s_parameters.loc[(data_frame_s_parameters["Size"]==3)|(data_frame_s_parameters["Size"]==4),"Size_new"]=int(1)
    data_frame_s_parameters["Size_new"] = data_frame_s_parameters["Size_new"].astype(int)


    additional_info_["Density"] = data_frame_s_parameters["Density"]
    additional_info_["Size"] = data_frame_s_parameters["Size"]
    additional_info_["Age"] = data_frame_s_parameters["Age"]
    #
    # Let me take out the size and density!
    del  data_frame_s_parameters["Size"]
    del  data_frame_s_parameters["Density"]
    del data_frame_s_parameters["PatientName"]

    return data_frame_s_parameters, additional_info_

data_frame_s_parameters, additional_info_ = preprocess_(data_frame_s_parameters)



def feature_selection_distribution_visualization(data_frame_s_parameters):

    data_frame_s_parameters.to_excel("C:/Users/ergLab/OneDrive/MITOS/MITOS/feature_vec.xlsx")
    # Let me look at the distributions of the s-parameters!
    #
    for each in range(0,38):
        data_frame_s_parameters["Frequency {}".format(each)] = np.log2(data_frame_s_parameters["Frequency {}".format(each)])
        sns.distplot(data_frame_s_parameters["Frequency {}".format(each)])
        plt.show()
    #
    #
    # 0-> Healthy, 1-> Benign, 2-> Malignant
    healthy = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==0]
    benign = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==1]
    malignant = data_frame_s_parameters[data_frame_s_parameters["Pathology"]==2]
    #
#    Let me check the age distributions!
    #
    bins = np.linspace(0, 200, 40)

    plt.hist(healthy["Age"], bins, label="healthy")
    plt.hist(benign["Age"], bins, label="benign")
    plt.hist(malignant["Age"], bins, label="malignant")
    plt.xlabel("Ages",fontsize=18)
    plt.ylabel("Frequency",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.show()
    #
#    Let me check breast size distribution!
    #
    size_healthy_s = len(healthy[healthy["Size_new"]==0])
    size_healthy_b =len(healthy)-len(healthy[healthy["Size_new"]==0])
    size_benign_s = len(benign[benign["Size_new"] == 0])
    size_benign_b = len(benign) - len(benign[benign["Size_new"] == 0])
    size_malignant_s = len(malignant[malignant["Size_new"] == 0])
    size_malignant_b = len(malignant) - len(malignant[malignant["Size_new"] == 0])
    #
    # create data
    x = ['Healthy', 'Benign', 'Malignant']
    small = [size_healthy_s,size_benign_s,size_malignant_s]
    big = [size_healthy_b,size_benign_b,size_malignant_b]


    # plot bars in stack manner
    plt.bar(x, small, color='orange',label="Small Breast")
    plt.bar(x, big, bottom=small, color='purple',label="Big Breast")

    plt.ylabel("Number of Samples",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)

    plt.show()



    # Let me check the Breast Side Distribution
    side_healthy_l = len(healthy[healthy["Side"]==0])
    side_healthy_r =len(healthy)-len(healthy[healthy["Side"]==0])
    side_benign_l = len(benign[benign["Side"] == 0])
    side_benign_r = len(benign) - len(benign[benign["Side"] == 0])
    side_malignant_l = len(malignant[malignant["Side"] == 0])
    side_malignant_r = len(malignant) - len(malignant[malignant["Side"] == 0])



    # create data for side distribution
    x = ['Healthy', 'Benign', 'Malignant']
    left = [side_healthy_l,side_benign_l,side_malignant_l]
    right = [side_healthy_r,side_benign_r,side_malignant_r]


    # plot bars in stack manner
    plt.bar(x, left, color='orange',label="Left Breast")
    plt.bar(x, right, bottom=left, color='purple',label="Right Breast")
    plt.ylabel("Number of Samples",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)

    plt.show()

    # return
# feature_selection_distribution_visualization(data_frame_s_parameters)
#

def  check_normality_in_continuous_variables(data_frame_s_parameters):
#    Let me check the normality with shapiro wilk test!
    for each in range(0,38):
        freq0_n = stats.shapiro(data_frame_s_parameters["Frequency {}".format(each)])[1]
        if freq0_n>=0.05:
            data_frame_s_parameters["Frequency {}".format(each)] = np.log10(data_frame_s_parameters["Frequency {}".format(each)])
            if stats.shapiro(data_frame_s_parameters["Frequency {}".format(each)])[1]>=0.05:
                print("S-parameters for frequency {} is not normally distributed!".format(each))

    # Let me check the normality in ages!
    if stats.shapiro(data_frame_s_parameters["Age"])[1]>=0.05:
        data_frame_s_parameters["Age"] = np.log10(data_frame_s_parameters["Age"])

    data_corr = data_frame_s_parameters.corr(method="pearson")

    # for each in data_frame_s_parameters.columns:
    #     corr_

    return data_frame_s_parameters

data_frame_s_parameters = check_normality_in_continuous_variables(data_frame_s_parameters)


def check_correlation_class_label(data_frame_s_parameters):

    freqs_to_remove = []
    coefs_pval =[]

    for each in range(0,38):

        coeff = stats.pointbiserialr(data_frame_s_parameters["Frequency {}".format(each)],
                                 data_frame_s_parameters["Pathology"])[1]

        coefs_pval.append(coeff)

        # Let me filter the values by correlation, when we take the correlation p-values<0.05
        if coeff>=0.1:
            freqs_to_remove.append("Frequency {}".format(each))

    # Now, let me plot the point biserial correlations of each frequency with the pathology of the sample!
    fs = []
    for each in range(0,38):
        fs.append("f{}".format(each))


    plt.scatter(fs,coefs_pval,marker=".")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Frequencies",fontsize=18)
    plt.ylabel("Point Biserial Correlation p-values",fontsize=18)
    plt.show()

    count=0
    my_freq_dict_pval ={}
    for each in range(0,38):
        my_freq_dict_pval["Frequency {}".format(each)]=coefs_pval[count]

        count=count+1

    data_frame_s_parameters.drop(freqs_to_remove, axis=1, inplace=True)

    return freqs_to_remove,my_freq_dict_pval

freqs_to_remove,my_freq_dict_pval = check_correlation_class_label(data_frame_s_parameters)



def check_correlation_between_attributes_(data_frame_s_parameters,my_freq_dict_pval):

    # Pearson correlation or LDA between attributes!

    # Let me try the pearson correlation!

    corr = data_frame_s_parameters.corr(method="pearson")
    correlate=corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    sns.heatmap(correlate,cmap="YlGnBu",annot=True)
    plt.show()

    # Let me put the coeff cutoff of 0.7, and find the attributes, which are similar to each other

    list_correlation_remove = []
    for each in range(0,len(corr)):
        counter=0
        for every in corr[corr.columns[each]]:
            sub=[]
            if every>0.7:
                sub.append([corr.columns[each]][0])
                sub.append(corr.index[counter])
                list_correlation_remove.append(sub)
            counter=counter+1

    count = 0
    for every in list_correlation_remove:
        if every[0]==every[1]:
            list_correlation_remove.pop(count)
        count=count+1



    # Let me remove one of the columns that are similar to each other and have lower p value, when doing the correlation against the labels!


    for each in list_correlation_remove:
        if "Age" not in each:
            if "Pathology" not in each:
                if "Side" not in each:
                    if "Size_new" not in each:
                        if my_freq_dict_pval[each[0]]>my_freq_dict_pval[each[1]]:
                            if each[0] in data_frame_s_parameters.columns:
                                data_frame_s_parameters.drop([each[0]],axis=1,inplace=True)
                        if my_freq_dict_pval[each[0]] < my_freq_dict_pval[each[1]]:
                            if each[1] in data_frame_s_parameters.columns:
                                data_frame_s_parameters.drop([each[1]], axis=1,inplace=True)

    print("Total number of the attributes is {}".format(len(data_frame_s_parameters.columns)-1))

    return data_frame_s_parameters

data_frame_s_parameters = check_correlation_between_attributes_(data_frame_s_parameters,my_freq_dict_pval)
print(data_frame_s_parameters)

# Now, let me do the scaling, we need to scale our continuous variables to have them in a certain scales!

def scaler(data_frame_s_parameters):

    print(data_frame_s_parameters[data_frame_s_parameters.columns[:-3]])
    # Let me scale the s-parameters and age!

    scale = StandardScaler()
    data_frame_s_parameters[data_frame_s_parameters.columns[:-3]] = scale.fit_transform(data_frame_s_parameters[data_frame_s_parameters.columns[:-3]])

    # data_frame_s_parameters = shuffle(data_frame_s_parameters)
    # data_frame_s_parameters.reset_index(inplace=True)
    # data_frame_s_parameters = data_frame_s_parameters[data_frame_s_parameters.columns[1:]]



    # Let me get the attributes from the parameters!

    Attribute_ = data_frame_s_parameters[data_frame_s_parameters.columns[:-2]]
    print(data_frame_s_parameters.columns)

    Attribute_["Size"] = data_frame_s_parameters["Size_new"]
    print(Attribute_.columns)
    print(Attribute_)
    Attribute_ = np.array(Attribute_)

    Label_ = data_frame_s_parameters["Pathology"]
    Label_ = np.array(Label_)



    return Attribute_,Label_

Attribute_,Label_ = scaler(data_frame_s_parameters)

## Let me now, try neural network classifications!

def classification_model(Attribute_,Label_,additional_info):

    model_name = "MLP"

    clf_ = neural_network.MLPClassifier(activation="relu",
                                           solver="sgd",
                                           learning_rate="constant",
                                           random_state=8)

    k_folds = KFold(n_splits=4,shuffle=True)

    scores_mlp = cross_val_score(clf_, Attribute_, Label_, cv=k_folds)

    print(model_name + " Cross Validation Scores: ", scores_mlp)
    print(model_name + " Average CV Score: ", scores_mlp.mean())
    print(model_name + " Number of CV Scores used in Average: ", len(scores_mlp))

    # Let me draw the confusion matrix!
    y_predict = cross_val_predict(clf_,Attribute_,Label_,cv=4)
    probability = cross_val_predict(clf_,Attribute_,Label_,cv=4,method="predict_proba")
    print(probability)

    data_att = pd.DataFrame(Attribute_)
    data_att["Size"] = additional_info["Size"]
    data_att["Age"] = additional_info["Age"]
    data_att["Density"] = additional_info["Density"]
    data_att["Label"] = Label_
    print(data_att)
    data_att["Probabilities of Malignancy"] = list_probabas_ = [i[1] for i in probability]
    data_att["Predict"] =y_predict

    conf_matrix = confusion_matrix(Label_, y_predict)
    sensitivity = (conf_matrix[1][1])/(conf_matrix[1][1]+conf_matrix[1][0])
    specifity = (conf_matrix[0][0])/(conf_matrix[0][0]+conf_matrix[0][1])
    best_point = [1-specifity,sensitivity]


    labels = Label_
    p = np.array(data_att["Probabilities of Malignancy"])
    fpr, tpr, _ = roc_curve(labels, p)
    auc = roc_auc_score(labels, p)

    ax = plt.subplot(111)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=18)
    plt.title(model_name + 'ROC Curve for the whole data', fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc, fontsize=18)
    plt.fill_between(fpr, tpr, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0], best_point[1], alpha=0.8, marker="o", color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.show()


    # Let me draw the confusion matrix according to age!

    age_less_than_40 = data_att[data_att["Age"]<=39]
    age_more_than_39 = data_att[data_att["Age"]>39]
    density_A_B = data_att[(data_att["Density"]=="a")|(data_att["Density"]=="b")]
    density_C_D = data_att[(data_att["Density"]=="c")|(data_att["Density"]=="d")]
    size_1_2 = data_att[(data_att["Size"]==1)|(data_att["Size"]==2)]
    size_3_4 = data_att[(data_att["Size"]==3)|(data_att["Size"]==4)]

    conf_matrix_by_age_lessthan_40 = confusion_matrix(np.array(age_less_than_40["Label"]), np.array(age_less_than_40["Predict"]))
    sensitivity = (conf_matrix_by_age_lessthan_40[1][1])/(conf_matrix_by_age_lessthan_40[1][1]+conf_matrix_by_age_lessthan_40[1][0])
    specifity = (conf_matrix_by_age_lessthan_40[0][0])/(conf_matrix_by_age_lessthan_40[0][0]+conf_matrix_by_age_lessthan_40[0][1])
    best_point = [1-specifity,sensitivity]

    sns.heatmap(conf_matrix_by_age_lessthan_40, annot=True, cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Young Patient Group", fontsize=18)
    plt.show()

    # Build a ROC Curve for age less than 40!!


    attributes_age_less40 = np.array(age_less_than_40[age_less_than_40.columns[:-6]])
    labels_less40 = np.array(age_less_than_40["Label"])
    p_less40 =  np.array(age_less_than_40["Probabilities of Malignancy"])
    fpr_less_40, tpr_less_40, _ = roc_curve(labels_less40, p_less40)
    auc_age_less_40 = roc_auc_score(labels_less40, p_less40)

    ax = plt.subplot(111)
    plt.plot(fpr_less_40, tpr_less_40)
    plt.ylabel('True Positive Rate (Sensitivity)',fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)',fontsize=18)
    plt.title(model_name + ' ROC Curve for the Young Patient Group',fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_age_less_40,fontsize=18)
    plt.fill_between(fpr_less_40, tpr_less_40, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0],best_point[1],alpha=0.8, marker="o",color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
    plt.show()


    conf_matrix_by_age_more_than_39 = confusion_matrix(np.array(age_more_than_39["Label"]), np.array(age_more_than_39["Predict"]))
    sensitivity = (conf_matrix_by_age_more_than_39[1][1])/(conf_matrix_by_age_more_than_39[1][1]+conf_matrix_by_age_more_than_39[1][0])
    specifity = (conf_matrix_by_age_more_than_39[0][0])/(conf_matrix_by_age_more_than_39[0][0]+conf_matrix_by_age_more_than_39[0][1])
    best_point = [1-specifity,sensitivity]


    sns.heatmap(conf_matrix_by_age_more_than_39,annot=True,cmap="YlGnBu",cbar=False)
    plt.xlabel("Predicted Label",fontsize=18)
    plt.ylabel("True Label",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Old Patient Group",fontsize=18)
    plt.show()


    labels_morethan39 = np.array(age_more_than_39["Label"])
    p_more40 =  np.array(age_more_than_39["Probabilities of Malignancy"])
    fpr_more39, tpr_more_39, _ = roc_curve(labels_morethan39, p_more40)
    auc_age_more_39 = roc_auc_score(labels_morethan39, p_more40)


    ax = plt.subplot(111)
    plt.plot(fpr_more39, tpr_more_39)
    plt.ylabel('True Positive Rate (Sensitivity)',fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)',fontsize=18)
    plt.title(model_name + ' ROC Curve for the Old Patient Group',fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_age_more_39,fontsize=18)
    plt.fill_between(fpr_more39, tpr_more_39, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0],best_point[1],alpha=0.8, marker="o",color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
    plt.show()


    conf_matrix_by_density_A_B = confusion_matrix(np.array(density_A_B["Label"]), np.array(density_A_B["Predict"]))
    sensitivity = (conf_matrix_by_density_A_B[1][1])/(conf_matrix_by_density_A_B[1][1]+conf_matrix_by_density_A_B[1][0])
    specifity = (conf_matrix_by_density_A_B[0][0])/(conf_matrix_by_density_A_B[0][0]+conf_matrix_by_density_A_B[0][1])
    best_point = [1-specifity,sensitivity]

    sns.heatmap(conf_matrix_by_density_A_B,annot=True,cmap="YlGnBu",cbar=False)
    plt.xlabel("Predicted Label",fontsize=18)
    plt.ylabel("True Label",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Non-dense Breasts",fontsize=18)
    plt.show()


    labels_density_A_B = np.array(density_A_B["Label"])
    p_nondense = np.array(density_A_B["Probabilities of Malignancy"])
    fpr_nondense, tpr_nondense, _ = roc_curve(labels_density_A_B, p_nondense)
    auc_nondense = roc_auc_score(labels_density_A_B, p_nondense)

    ax = plt.subplot(111)
    plt.plot(fpr_nondense, tpr_nondense)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=18)
    plt.title(model_name + ' ROC Curve for the Non-dense breasts', fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_nondense, fontsize=18)
    plt.fill_between(fpr_nondense, tpr_nondense, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0], best_point[1], alpha=0.8, marker="o", color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.show()

    conf_matrix_by_density_C_D = confusion_matrix(np.array(density_C_D["Label"]), np.array(density_C_D["Predict"]))
    sensitivity = (conf_matrix_by_density_C_D[1][1])/(conf_matrix_by_density_C_D[1][1]+conf_matrix_by_density_C_D[1][0])
    specifity = (conf_matrix_by_density_C_D[0][0])/(conf_matrix_by_density_C_D[0][0]+conf_matrix_by_density_C_D[0][1])
    best_point = [1-specifity,sensitivity]

    sns.heatmap(conf_matrix_by_density_C_D,annot=True,cmap="YlGnBu",cbar=False)
    plt.xlabel("Predicted Label",fontsize=18)
    plt.ylabel("True Label",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Dense Breasts",fontsize=18)
    plt.show()


    labels_density_C_D = np.array(density_C_D["Label"])
    p_dense = np.array(density_C_D["Probabilities of Malignancy"])
    fpr_dense, tpr_dense, _ = roc_curve(labels_density_C_D, p_dense)
    auc_dense = roc_auc_score(labels_density_C_D, p_dense)

    ax = plt.subplot(111)
    plt.plot(fpr_dense, tpr_dense)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=18)
    plt.title(model_name + ' ROC Curve for the dense breasts', fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_dense, fontsize=18)
    plt.fill_between(fpr_dense, tpr_dense, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0], best_point[1], alpha=0.8, marker="o", color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.show()



    conf_matrix_by_size_1_2 = confusion_matrix(np.array(size_1_2["Label"]), np.array(size_1_2["Predict"]))
    sensitivity = (conf_matrix_by_size_1_2[1][1])/(conf_matrix_by_size_1_2[1][1]+conf_matrix_by_size_1_2[1][0])
    specifity = (conf_matrix_by_size_1_2[0][0])/(conf_matrix_by_size_1_2[0][0]+conf_matrix_by_size_1_2[0][1])
    best_point = [1-specifity,sensitivity]
    sns.heatmap(conf_matrix_by_size_1_2, annot=True, cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Small Breasts", fontsize=18)
    plt.show()

    labels_size_1_2 = np.array(size_1_2["Label"])
    p_small = np.array(size_1_2["Probabilities of Malignancy"])
    fpr_small, tpr_small, _ = roc_curve(labels_size_1_2, p_small)
    auc_small = roc_auc_score(labels_size_1_2, p_small)

    ax = plt.subplot(111)
    plt.plot(fpr_small, tpr_small)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=18)
    plt.title(model_name + ' ROC Curve for the small breasts', fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_small, fontsize=18)
    plt.fill_between(fpr_small, tpr_small, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0], best_point[1], alpha=0.8, marker="o", color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.show()



    conf_matrix_by_size_3_4 = confusion_matrix(np.array(size_3_4["Label"]), np.array(size_3_4["Predict"]))
    sensitivity = (conf_matrix_by_size_3_4[1][1])/(conf_matrix_by_size_3_4[1][1]+conf_matrix_by_size_3_4[1][0])
    specifity = (conf_matrix_by_size_3_4[0][0])/(conf_matrix_by_size_3_4[0][0]+conf_matrix_by_size_3_4[0][1])
    best_point = [1-specifity,sensitivity]

    sns.heatmap(conf_matrix_by_size_3_4, annot=True, cmap="YlGnBu",cbar=False)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(model_name + " Confusion Matrix Large Breasts", fontsize=18)
    plt.show()


    labels_size_3_4 = np.array(size_3_4["Label"])
    p_large = np.array(size_3_4["Probabilities of Malignancy"])
    fpr_large, tpr_large, _ = roc_curve(labels_size_3_4, p_large)
    auc_large = roc_auc_score(labels_size_3_4, p_large)

    ax = plt.subplot(111)
    plt.plot(fpr_large, tpr_large)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=18)
    plt.title(model_name + ' ROC Curve for the large breasts', fontsize=18)
    plt.text(0.5, 0.4, 'AUC = %.2f' % auc_large, fontsize=18)
    plt.fill_between(fpr_large, tpr_large, color='lightskyblue')
    plt.plot([0, 1], [0, 1], color='r', linestyle='-.', linewidth=0.5, label='Random Classifier')
    plt.scatter(best_point[0], best_point[1], alpha=0.8, marker="o", color="r")
    plt.legend(loc='lower right')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.show()





    return #scores_mlp

classification_model(Attribute_,Label_,additional_info_)
