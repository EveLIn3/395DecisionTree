# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:45:27 2018

@author: dian & rin
"""

import scipy.io
import numpy as np
import dTree

from sklearn.model_selection import KFold

mat = scipy.io.loadmat('cleandata_students.mat')

example_clean = mat['x']
label_clean = mat['y']

attributes = set()
for a in range(45):
    attributes.add(a)

binary_labels = np.zeros((len(example_clean), 6))
for i in range(1, 7):
    binary = (label_clean == i).astype(int)[:, 0]
    binary_labels[:, i - 1] = binary


def classifier(tree, x_test):
    if tree.kids == []:
        return tree.label
    else:
        left = tree.kids[0]
        right = tree.kids[1]
        if x_test[tree.op] == 1:
            label = classifier(left, x_test)
        else:
            label = classifier(right, x_test)
        return label


'''
Evaluation
'''


def confusion_matrix(prediction, label):
    confusion_matrix = np.zeros((2, 2))
    k = len(prediction.T)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(k):
        # true
        T = len(prediction[prediction == label, i])
        # false
        F = len(prediction[prediction == label, i])
        # true positive
        TP = prediction[label == 1, i].sum() + TP
        # true negative
        TN = (T - TP) + TN
        # false positive
        FP = len(prediction[label == 1, i]) - TP
        # false nagtive
        FN = F - FP
    confusion_matrix = np.array([TP, TN], [FP, TN])
    return confusion_matrix


def recall_rate(prediction, binary_label):
    # TP + FN
    label_p = binary_label[binary_label == 1].sum()
    # true positive
    if label_p == 0:
        return 1.0
    TP = prediction[binary_label == 1].sum()
    # recall rate
    recall_rate = TP / label_p

    return recall_rate


def precision_rate(prediction, binary_label):
    # TP + FP
    pre_p = prediction[prediction == 1].sum()
    if pre_p == 0:
        return 1.0
        # true positive
    TP = prediction[binary_label == 1].sum()
    # precision rate
    precision_rate = TP / pre_p

    return precision_rate


def F_1(ave_recall, ave_precision):
    # F_alpha measure,alpha =1
    F_alpha = (1 + 1) * (ave_precision * ave_recall) / (1 * ave_precision + ave_recall)
    return F_alpha


'''
k-fold cross validation
k = 10
'''
kf = KFold(n_splits=10)
'''
emotion ={0,1,2,3,4,5}

'''


def k_th_cross_validation(examples, labels, kfold=10):
    num_examples = np.shape(examples)[0]
    num_each_fold = int(num_examples / kfold)
    accuracy_list = []
    for k in range(kfold):
        X_test = examples[k*num_each_fold:(k+1)*num_each_fold, :]
        X_train = np.delete(examples, np.s_[k*num_each_fold:(k+1)*num_each_fold], axis=0)
        Y_test = labels[k*num_each_fold:(k+1)*num_each_fold]
        Y_train = np.delete(labels, np.s_[k*num_each_fold:(k+1)*num_each_fold], axis=0)

        trained_trees_list = []
        for i in range(1,7):
            attributes = set()
            for a in range(45):
                attributes.add(a)

            train_binary_labels = (Y_train == i).astype(int)
            #	print("=====================Labels Length {}".format(np.shape(binary_labels)))
            #	print(np.sum(binary_labels))
            trained_tree = dTree.decision_tree_learning(X_train, attributes, train_binary_labels)
            trained_trees_list.append(trained_tree)

        X_test_list = X_test.tolist()
        Y_predict_list = [dTree.testTrees(trained_trees_list, x) for x in X_test_list]
        Y_predict = np.reshape(np.array(Y_predict_list), (len(Y_predict_list), 1))
        correct_prediction = np.sum(Y_test == Y_predict)
        print(Y_predict_list)
        # print(correct_prediction)
        accuracy = correct_prediction / num_each_fold
        accuracy_list.append(accuracy)

    print(accuracy_list)


def cross_validation(examples, emotion, binary_labels, attributes):
    recall = np.zeros((10, 1), float)
    precision = np.zeros((10, 1), float)
    index = 0
    for train_index, test_index in kf.split(examples):
        #            print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = examples[train_index], example_clean[test_index]
        y_train, y_test = binary_labels[:, emotion][train_index], binary_labels[:, emotion][test_index]
        # generate the decision tree
        trained_trees = dTree.decision_tree_learning(X_train, attributes, y_train)
        # classify test data
        # X_prediction = np.zeros((X_test.shape[0], 1))
        X_prediction = [dTree.testTrees(trained_trees, x) for x in X_test]
        # prediction = append(prediction,X_prediction,axis = 1)

        for i in range(X_test.shape[0]):
            X_prediction[i] = classifier(tree, X_test[i])

        recall[index] = recall_rate(X_prediction, y_test)
        precision[index] = precision_rate(X_prediction, y_test)
        index = index + 1

    # print("shape recall {}".format(np.shape(recall)))
    # print("shape precision {}".format(np.shape(precision)))
    # print("recall ", recall)
    # print("precision ", precision)

    ave_recall = np.sum(recall, axis=1)
    ave_precision = np.sum(precision, axis=1)
    f1 = F_1(ave_recall, ave_precision)

    return ave_recall, ave_precision, f1


'''
Cross Validation
'''

# -----------------------------------------------------------------------------------

# cross_validation(example_clean, 1, binary_labels, attributes)

k_th_cross_validation(example_clean, label_clean)