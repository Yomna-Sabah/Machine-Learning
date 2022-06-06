import numpy as np
import math


class Node():
    """Contains the information of the node and other nodes of the Decision Tree."""

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier():
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self):
        self.node = None
        

    def get_entropy(self, x_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        __________
        :return: entropy: float, Entropy.
        """
        # sorted labels by instance id
        labels = [self.labels[i] for i in x_ids]

        # count number of instances of each category
        label_count = [labels.count(x) for x in self.labelCategories]

        # calculate the entropy for each category and sum them --> using entropy equation
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        # return calculated entropy
        return entropy

    def get_information_gain(self, x_ids, feature_id):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param feature_id: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        info_gain = self.get_entropy(x_ids)
        
        # store in a list all the values of the chosen feature
        x_features = [self.X[x][feature_id] for x in x_ids]

        # get unique values
        feature_vals = list(set(x_features))

        # get frequency of each value
        feature_vals_count = [x_features.count(x) for x in feature_vals]

        # get the feature values ids
        feature_vals_id = [[x_ids[i]for i, x in enumerate(x_features)if x == y]for y in feature_vals]

        # compute the information gain with the chosen feature --> using information gain equation.
        info_gain = info_gain - sum([val_counts / len(x_ids) * self.get_entropy(val_ids)for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        # return calculated information gain of given feature
        return info_gain

    def get_feature_max_information_gain(self, x_ids, feature_ids):
        """Finds the feature that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        __________
        :returns: string and int, feature name and feature id of the feature that maximizes the information gain
        """
        # get the entropy for each feature
        features_entropy = [self.get_information_gain(x_ids, feature_id) for feature_id in feature_ids]

        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]

        # return feature name and id that maximize the information gain
        return self.feature_names[max_id], max_id

    def fit(self, X, y):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.
        Parameters
        __________
        :param X: DataFrame, DataFrame containing set of data to train model.
        :param y: DataFrame, DataFrame containing set of true values of this data to train model.
        __________
        :return: None
        """

        # name of features
        self.feature_names = list(X.keys())

        # name of target feature
        self.feature_label = list(y.columns)
        self.feature_label = self.feature_label[0]

        # features
        self.X = np.array(X)

        # target labels
        self.labels = np.array(y)
        self.labels = self.labels[:,0]
        # target label category
        self.labelCategories = list(set(self.labels))
        # number of instances of each category
        self.labelCategoriesCount = [list(self.labels).count(x) for x in self.labelCategories]
        
        # calculates the initial entropy
        self.entropy = self.get_entropy([x for x in range(len(self.labels))])

        # assign unique number to each instance
        x_ids = [x for x in range(len(self.X))]

        # assign unique number to each feature
        feature_ids = [x for x in range(len(self.feature_names))]

        # define node variable - instance of the class Node
        self.node = self.fit_recv(x_ids, feature_ids, self.node)

    def fit_recv(self, x_ids, feature_ids, node):
        """ID3 algorithm. It is called recursively until some criteria is met.
        Parameters
        __________
        :param x_ids: list, list containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        :param node: object, An instance of the class Nodes
        __________
        :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
        """
        if not node:
            # initialize nodes
            node = Node()

        # sorted labels by instance id
        labels_in_features = [self.labels[x] for x in x_ids]

        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node
        
        # if there are not more feature to compute, return node with the most probable class
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            return node
        
        # else...
        # choose the feature that maximizes the information gain
        best_feature_name, best_feature_id = self.get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []

        # value of the chosen feature for each instance
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))

        # loop through all the values
        for value in feature_values:
            child = Node()

            # add a branch from the node to each feature value in our feature
            child.value = value

            # append new child node to current node
            node.childs.append(child)

            # instance that take the brach
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]

            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)

            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                # recursively call the algorithm
                child.next = self.fit_recv(child_x_ids, feature_ids, child.next)

        # return instance of the class Node containing all the information of the nodes in the Decision Tree
        return node

    def accuracy_score(self, X, y):
        """ Calculates the accuracy score of a set of data.
        Parameters
        __________
        :param X: DataFrame, DataFrame containing set of data.
        :param y: DataFrame, DataFrame containing set of true values of this data.
        __________
        :return: Accuracy: float, Accuracy.
        """
        # initialize counter for true predicted instances
        correct_counter = 0
        total = 0
        
        # loop through all instances in data
        for index,instance in X.iterrows():
            # get y_true to of the index of this instance to compare with
            y_true = list(y.loc[[index]][self.feature_label])
            y_true = y_true[0]
            total += 1

            # call predict function to get predicted value of this instance
            y_pred = self.predict(instance)

            # compare predicted value by true value, if equal increment counter of correct predicted instances 
            if y_pred == y_true:
                correct_counter += 1
        
        # calculate accuracy score of set of data
        accuracy_score = float(correct_counter) / float(total)

        # return accuracy score of all instances
        return accuracy_score


    def predict(self,instance):
        """ Calculates the predicted vlaue of a row of data (One Instance).
        Parameters
        __________
        :param instance: row of DataFrame, containing data of row.
        __________
        :return: predicted_value: predicted value of this instance.
        """
        # get instance of the class Node containing all the information of the nodes in the Decision Tree 
        Node = self.node

        # loop until there no child for this node
        while Node.childs != None:

            # get value of current node
            node_value = Node.value

            # get childs of current node
            node_child = Node.childs

            # loop for childs of current node
            for node in node_child:

                # compare instance of node value (feature name) by the node value
                if instance[node_value] == node.value:
                    # set next node to be the current node
                    Node = node.next

        # value of final node is the predicted value
        predicted_value = Node.value

        # return predicted value of this instance
        return predicted_value
