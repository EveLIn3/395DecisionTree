# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:45:27 2018

@author: dian & rin
"""

import scipy.io
import numpy as np
import dTree

mat = scipy.io.loadmat('noisydata_students.mat')

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

'''
noClass = 6
'''


def confusion_matrix(prediction, actLabel, noClass=6):
    confusion_matrix = np.zeros((noClass, noClass))
    for i in range(noClass):
        for j in range(noClass):
            # pre_pos = prediction[prediction == j+1]
            confusion_matrix[i, j] = len(prediction[np.logical_and(prediction == j+1, actLabel == i+1)])
    return confusion_matrix

    
def recall_rate(confusion_matrix):
    k = len(confusion_matrix)
    recall_rate = np.zeros((1,k))
    for i in range(k):
        TP = confusion_matrix[i,i]
        TPplusFN = confusion_matrix[i,:].sum()
        recall_rate[:,i] = TP/TPplusFN
    return recall_rate


def precision_rate(confusion_matrix):
    k = len(confusion_matrix)
    precision_rate = np.zeros((1,k))
    for i in range(k):
        TP = confusion_matrix[i,i]
        TPplusFP = confusion_matrix[:,i].sum()
        precision_rate[:,i] = TP/TPplusFP
    return precision_rate


def F_1(ave_recall, ave_precision):
    # F_alpha measure,alpha =1
    F_alpha = (1 + 1) * (ave_precision * ave_recall) / (1 * ave_precision + ave_recall)
    return F_alpha


'''
k-fold cross validation
k = 10
'''
# kf = KFold(n_splits=10)
'''
emotion ={0,1,2,3,4,5}

'''


def k_th_cross_validation(examples, labels, kfold=10):
    num_examples = np.shape(examples)[0]
    num_each_fold = int(num_examples / kfold)
    accuracy_list = []
    confusion_matrix_total = np.zeros((6,6))
    print("Running cross validation...")
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
            # print("=====================Labels Length {}".format(np.shape(binary_labels)))
            # print(np.sum(binary_labels))
            trained_tree = dTree.decision_tree_learning(X_train, attributes, train_binary_labels)
            trained_trees_list.append(trained_tree)

        X_test_list = X_test.tolist()
        Y_predict_list = [dTree.testTrees_depth_determined(trained_trees_list, x) for x in X_test_list]
        Y_predict = np.reshape(np.array(Y_predict_list), (len(Y_predict_list), 1))
        
        # confusion_matrix1 = np.zeros((6,6))
        
        correct_prediction = np.sum(Y_test == Y_predict)
        print("Fold {}".format(k+1))
        # print(correct_prediction)
        accuracy = correct_prediction / num_each_fold
        accuracy_list.append(accuracy)

        # confusion_matrix1 = np.zeros((6,6))
        confusion_matrix1 = confusion_matrix(Y_predict, Y_test, 6)
        confusion_matrix_total = confusion_matrix_total + confusion_matrix1
    
    ave_recall_rate = recall_rate(confusion_matrix_total)
    ave_precision_rate = precision_rate(confusion_matrix_total)
    ave_f1_value = F_1(ave_recall_rate,ave_precision_rate)

    print()
    print("Confusion Matrix:")
    print(confusion_matrix_total)
    print("{} Fold Cross Validation Avg Accuracy: {:.4f}".format(kfold, sum(accuracy_list) / kfold))
    print("Variance: {:.5f}".format(np.var(accuracy_list)))

    print()
    print("Emotion           1           2           3           4           5           6")
    print("Precision: {}".format(ave_precision_rate[0]))
    print("Recall:    {}".format(ave_recall_rate[0]))
    print("F-1:       {}".format(ave_f1_value[0]))

'''
Cross Validation
'''

k_th_cross_validation(example_clean, label_clean)