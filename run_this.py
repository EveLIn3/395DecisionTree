import scipy.io
import numpy as np
import dTree
from dTree import Tree  # necessary for recognizing class object
import pickle
import tree_plotter

mat = scipy.io.loadmat('cleandata_students.mat')  # please modify the file path here

examples_aus = mat['x']
true_labels = mat['y']

with open('saved_trees.pkl', 'rb') as saved_trees_file:
    saved_trees_list = pickle.load(saved_trees_file)

for i in range(len(saved_trees_list)):
    print("Decision tree for label No.{}".format(i+1))
    tree_plotter.print_tree(saved_trees_list[i], 'kids', 'op')
    print("==================================")

X_test_list = examples_aus.tolist()
Y_predict_list = [dTree.testTrees_depth_determined(saved_trees_list, x) for x in X_test_list]
Y_predict = np.reshape(np.array(Y_predict_list), (len(Y_predict_list), 1))

correct_prediction = np.sum(true_labels == Y_predict)
accuracy = correct_prediction / len(X_test_list)

print("Please scroll up to check the quick text version of visualization.")
print("For full graphical visualization, please check the \'Instructions.txt\'.")
print()
print("The classification rate is: {:.4f}".format(accuracy))
