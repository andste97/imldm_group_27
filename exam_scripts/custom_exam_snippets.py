import math

import scipy.special

import toolbox_extended as te
import toolbox_02450 as tb
import numpy as np
import pandas as pd
from exam_toolbox import *
import re
import os


class exam:
    def create_dendrogram():
        data = [[0.0, 2.0, 5.7, 0.9, 2.9, 1.8, 2.7, 3.7, 5.3, 5.1],
                [2.0, 0.0, 5.6, 2.4, 2.5, 3.0, 3.5, 4.3, 6.0, 6.2],
                [5.7, 5.6, 0.0, 5.0, 5.1, 4.0, 3.3, 5.4, 1.2, 1.8],
                [0.9, 2.4, 5.0, 0.0, 2.7, 2.1, 2.2, 3.5, 4.6, 4.4],
                [2.9, 2.5, 5.1, 2.7, 0.0, 3.5, 3.7, 4.0, 5.8, 5.7],
                [1.8, 3.0, 4.0, 2.1, 3.5, 0.0, 1.7, 5.3, 3.8, 3.7],
                [2.7, 3.5, 3.3, 2.2, 3.7, 1.7, 0.0, 4.2, 3.1, 3.2],
                [3.7, 4.3, 5.4, 3.5, 4.0, 5.3, 4.2, 0.0, 5.5, 6.0],
                [5.3, 6.0, 1.2, 4.6, 5.8, 3.8, 3.1, 5.5, 0.0, 2.1],
                [5.1, 6.2, 1.8, 4.4, 5.7, 3.7, 3.2, 6.0, 2.1, 0.0]]

        df = pd.DataFrame(data)
        dendro = cluster()
        dendro.dendro_plot(df, "complete")

    def create_rand_jaccard():
        """Exam 2019, question 7:
        Consider dendrogram 1 from Figure 3.
        Suppose we apply a cutoff (indicated by the black line)
        thereby generating three clusters. We wish to compare
        the quality of this clustering, Q, to the ground-truth
        clustering, Z, indicated by the colors in Table 2. Recall
        the Jaccard similarity of the two clusters is
        [formula]
        in the notation of the lecture notes. What is the
        Jaccard similarity of the two clusterings?
        """
        truth = [0,0,1,1,1,2,2,2,2,2]
        clusters_to_compare_agains_truth = [0,0,2,0,0,0,0,2,2,1]
        #df = pd.DataFrame(data)
        similarity = cluster()
        similarity.cluster_similarity(clusters_to_compare_agains_truth, truth)

    def calc_impurity_gain():
        """calculates impurity measure of split according to hunts algorithm
        purity_mesaure can be one of: "gini", "class_error", "entropy" """
        dec = decision_trees()
        root = np.array([263, 359, 358])
        left = np.array([143, 137, 54])
        right = root - left
        dec.purity_gain(root, left, right, purity_measure="class_error")

    def calc_naive_bayes_3():
        data = [[0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0]]

        # class numbers for each row. Starts at zero.
        classes = [0,0,1,1,1,2,2,2,2,2]
        # the number of columns used for predictions. Starts at zero
        cols_used_for_pred = [1,3,4]
        # the numbers that are given in the assignment, ie p(x0=1, x2=0|y=1)...
        col_values = [0,1,0]
        # value of y in the term to predict
        class_to_predict = 1

        df = pd.DataFrame(data)
        bayes = supervised()
        bayes.naive_bayes(classes, df, cols_used_for_pred, col_values, class_to_predict)

    def similarity_between_columns():
        """calculates smc, jaccard and cos similarities"""
        data = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0]])

        #calculate similarity between different columns:
        sim = similarity()
        print("o1,o3")
        sim.measures(data[0,:], data[2,:])
        #o2,o3
        print("o2,o3")
        sim.measures(data[1, :], data[2, :])
        #o2,o4
        print("o2,o4")
        sim.measures(data[1, :], data[3, :])

    def adaBoost_new_weights():
        "simple adaboost usage, see documentation of adaboost method for infos"
        boost = adaboost()
        list_misclassified = [0,1,1,1,0,1,1]
        boost.adaboost(list_misclassified, 1)

    def multinomial_regression():
        "Exam 2019 Q11"
        # first number is given as 1 in the assignment, see y1 and y2
        point = np.asarray([1,-0,-1])

        wa1 = np.asarray([-0.77, -5.54, 0.01])
        wa2 = np.asarray([0.26, -2.09, -0.03])

        wb1 = np.asarray([0.51, 1.65, 0.01])
        wb2 = np.asarray([0.1, 3.8, 0.04])

        wc1 = np.asarray([-0.9, -4.39, -0.0])
        wc2 = np.asarray([-0.09, -2.45, -0.04])

        wd1 = np.asarray([-1.22, -9.88, -0.01])
        wd2 = np.asarray([-0.28, -2.9, -0.01])

        # y3 is always zero (as seen from the default solution)
        # choose maximum from each observation to predict class
        # if both predictions negative, choose y3 (as zero bigger than negative)
        print("A: ", point@wa1, " ", point@wa2)
        print("B: ", point @ wb1, " ", point @ wb2)
        print("C: ", point @ wc1, " ", point @ wc2)
        print("D: ", point @ wd1, " ", point @ wd2)

exam.create_dendrogram()
exam.create_rand_jaccard()
exam.calc_impurity_gain()
print("naive bayes:")
exam.calc_naive_bayes_3()
print("similarity:")
exam.similarity_between_columns()
print("adaboost new weights: ")
exam.adaBoost_new_weights()
print("results multinomial regression: ")
exam.multinomial_regression()


def exam2019_q21():
    point = np.array([3,4])
    regularizer = np.array([2,4])
    p1 = np.linalg.norm(point-regularizer,1)
    res = np.sum(p1)