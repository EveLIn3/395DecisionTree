import numpy as np
import scipy.io as sio
import tree_plotter
import random

# hyper-parameters    0.5 & 1 delivers 0.744
small_enough_entropy = 0.4
min_sample_per_node = 1


class Tree:
    def __init__(self, op, kids, label):
        self.op = op
        self.kids = kids
        self.label = label


def majority_value(binary_targets):
    N = binary_targets.shape[0]
    v = np.sum(binary_targets)
    if v >= N / 2.0:
        majority = 1
    else:
        majority = 0
    return majority


def choose_best_decision_attribute(example, attributes, binary_target):
    # information gain = p * log p
    attribute = 0
    N = example.shape[0]

    postiveN = np.sum(binary_target)
    # print('N ', N)
    # print('postiveN ', postiveN)
    propor = 1.0 * postiveN / N
    entropyE = -1 * ((propor) * np.log2(propor) + (1 - propor) * np.log2(1 - propor))

    if entropyE < small_enough_entropy:  # pruning tree node
        return -1

    prev_info = 0
    for attri in attributes:
        exampleA = example[:, attri]  # column with attribute = attri
        posAttr = np.sum(exampleA)
        # print("--------------")
        # print(np.shape(exampleA))
        atrP = 1.0 * posAttr / N

        # aPos = 0
        aNeg = 0
        # need to calculate entropy for each Sv
        value_one_target = binary_target[exampleA == 1]
        # totalOOL = one_one.shape[0]
        totalOOL = len(value_one_target)
        if totalOOL == 0:
            aPos = 0
        else:
            totalOOS = np.sum(value_one_target)
            totalOZS = totalOOL - totalOOS

            p1 = 1.0 * totalOOS / totalOOL  # 1 - 1
            p2 = 1.0 * totalOZS / totalOOL  # 1 - 0

            if p1 == 0:
                aPos = -1.0 * p2 * np.log2(p2)
            elif p2 == 0:
                aPos = -1.0 * p1 * np.log2(p1)
            else:
                aPos = -1.0 * (p2 * np.log2(p2) + p1 * np.log2(p1))

            value_zero_target = binary_target[exampleA == 0]
            totalZOL = value_zero_target.shape[0]

            if totalZOL == 0:
                aNeg = 0
            else:
                totalZOS = np.sum(value_zero_target)
                totalZZS = totalZOL - totalZOS
                p3 = 1.0 * totalZOS / totalZOL  # 0 - 1
                p4 = 1.0 * totalZZS / totalZOL  # 0 - 0

                if p3 == 0:
                    aNeg = -1.0 * p4 * np.log2(p4)
                elif p4 == 0:
                    aNeg = -1.0 * p3 * np.log2(p3)
                else:
                    aNeg = -1.0 * (p3 * np.log2(p3) + p4 * np.log2(p4))

        infoG = entropyE - atrP * aPos - (1.0 - atrP) * aNeg
        if prev_info < infoG:
            prev_info = infoG
            attribute = attri

    return attribute


def decision_tree_learning(examples, attributes, binary_targets):
    num_examples = np.shape(examples)[0]
    sum_labels = np.sum(binary_targets)
    if sum_labels == 0:
        return Tree(None, [], 0)
    elif sum_labels == num_examples:
        return Tree(None, [], 1)
    elif len(attributes) == 0:
        return Tree(None, [], majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        if best_attribute < 0:
            return Tree(None, [], majority_value(binary_targets))
        # print("Best Attr: {}".format(best_attribute))
        attribute_values = examples[:, best_attribute]

        attr_postive_index = attribute_values == 1
        attr_negative_index = attribute_values == 0

        attr_postive_examples = examples[attr_postive_index]
        attr_negative_examples = examples[attr_negative_index]
        # print("shape + {}".format(np.shape(attr_postive_examples)))
        # print("shape - {}".format(np.shape(attr_negative_examples)))
        attr_postive_label = binary_targets[attr_postive_index]
        attr_negative_label = binary_targets[attr_negative_index]

        num_attr_1_examples = np.shape(attr_postive_examples)[0]
        num_attr_0_examples = np.shape(attr_negative_examples)[0]

        if num_attr_1_examples < min_sample_per_node or num_attr_0_examples < min_sample_per_node:
            return Tree(None, [], majority_value(binary_targets))
        # if np.shape(attr_postive_examples)[0] == 0 or np.shape(attr_negative_examples)[0] == 0:
        #     return Tree(None, [], majority_value(binary_targets))
        else:
            tree = Tree(best_attribute, [], None)
            attributes.remove(best_attribute)
            attributes_copy = attributes.copy()  # use copy of set, so one attribute may be used at two sub-trees
            left_kid = decision_tree_learning(attr_postive_examples, attributes_copy, attr_postive_label)
            right_kid = decision_tree_learning(attr_negative_examples, attributes_copy, attr_negative_label)
            tree.kids = [left_kid, right_kid]
            return tree


def testTrees(trees, test_sample):
    predicted_label = 0
    num_of_classes = len(trees)
    for i in range(num_of_classes):
        root = trees[i]
        while len(root.kids) > 0:
            action_unit_index = root.op
            if test_sample[action_unit_index] == 1:
                # print("going left")
                root = root.kids[0]  # left kid
            else:
                # print("going right")
                root = root.kids[1]  # right kid

        predicted_class = root.label
        if predicted_class == 1:
            predicted_label = i + 1
            break
        elif i == num_of_classes - 2:
            predicted_label = i + 2

    return predicted_label


def testTrees_random(trees, test_sample):
    predicted_label = 0
    num_of_classes = len(trees)
    predictions_depth_dict = {}
    for i in range(num_of_classes):
        root = trees[i]
        depth = 0
        while len(root.kids) > 0:
            action_unit_index = root.op
            depth += 1
            if test_sample[action_unit_index] == 1:
                root = root.kids[0]  # left kid
            else:
                root = root.kids[1]  # right kid

        predicted_class = root.label
        if predicted_class == 1:
            predictions_depth_dict[i + 1] = depth
            predicted_label = i + 1

    if predicted_label == 0:
        return np.random.randint(6, size=1)[0]
    elif len(predictions_depth_dict) == 1:
        return predicted_label
    else:
        return random.choice(list(predictions_depth_dict.keys()))


def testTrees_depth_determined(trees, test_sample):
    predicted_label = 0
    num_of_classes = len(trees)
    predictions_depth_dict = {}
    final_depth_dict = {}
    for i in range(num_of_classes):
        root = trees[i]
        depth = 0
        while len(root.kids) > 0:
            action_unit_index = root.op
            depth += 1
            if test_sample[action_unit_index] == 1:
                root = root.kids[0]  # left kid
            else:
                root = root.kids[1]  # right kid

        predicted_class = root.label
        final_depth_dict[i+1] = depth
        if predicted_class == 1:
            predictions_depth_dict[i + 1] = depth
            predicted_label = i + 1

    if predicted_label == 0:
        shallowest_label = max(final_depth_dict, key=final_depth_dict.get)
        return shallowest_label
        # return np.random.randint(6, size=1)[0]
    elif len(predictions_depth_dict) == 1:
        return predicted_label
    else:
        shallowest_predict_label = min(predictions_depth_dict, key=predictions_depth_dict.get)
        return shallowest_predict_label


# Example = np.array([[1, 0], [1, 1], [1, 0], [1, 0], [0, 0], [0, 1], [0, 1],
#                     [1, 0], [0, 0], [0, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
# Atr = np.array([0, 1])
# binary = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
# k = choose_best_decision_attribute(Example, Atr, binary)
# print("k is ", k)

data = sio.loadmat('cleandata_students.mat')
features = data['x']
labels = data['y']
print("Visualizing Trees with all training examples...")

for i in range(1, 7):
    attributes = set()
    for a in range(45):
        attributes.add(a)

    binary_labels = (labels == i).astype(int)
    trained_tree = decision_tree_learning(features, attributes, binary_labels)
    print("======================================================================")
    print("Decision tree for label No.{}".format(i))
    tree_plotter.print_tree(trained_tree, childattr='kids', nameattr='op')
    print("======================================================================")
