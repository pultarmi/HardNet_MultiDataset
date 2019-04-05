"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.
"""

import numpy as np
import sklearn.metrics

def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array). 
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels)) 

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)

def AP(labels, distances):
    scores = 1.0 / (distances + 1e-8)
    # labels = labels[np.argsort(distances)]
    # print(labels[:20])
    # print(distances[np.argsort(distances)][:20])
    # print(distances)
    res = sklearn.metrics.average_precision_score(labels, scores)
    return res

def prec_recall_curve(labels, distances):
    scores = 1.0 / (distances + 1e-8)
    # labels = labels[np.argsort(distances)]
    # print(labels[:20])
    # print(distances[np.argsort(distances)][:20])
    # print(distances)
    res = sklearn.metrics.precision_recall_curve(labels, scores, pos_label=1)
    return res

def ErrorRateAt95Recall_AndMatchabilityAcc(labels, scores, m_a, m_p):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    idxs = np.argsort(distances)
    idxs_a = np.argsort(m_a)
    idxs_p = np.argsort(m_p)
    labels = labels[idxs]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array). 
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    fp_idxs = labels[:threshold_index] == 0
    FP = np.sum(fp_idxs) # Below threshold (i.e., labelled positive), but should be negative
    fp_m = ((m_a[idxs[:threshold_index]]) [fp_idxs])#.mean() 
    print('FP mean, std, ', fp_m.mean(), fp_m.std())
    tp_idxs = labels[:threshold_index] == 1
    TP = np.sum(tp_idxs) # Below threshold (i.e., labelled positive), and should be positive
    tp_m = ((m_a[idxs[:threshold_index]]) [tp_idxs])#.mean() 
    print('TP mean, std, ', tp_m.mean(), tp_m.std())

    tn_idxs = labels[threshold_index:] == 0
    TN = np.sum(tn_idxs) # Above threshold (i.e., labelled negative), and should be negative
    tn_m = ((m_a[idxs[threshold_index:]]) [tn_idxs])#.mean() 
    print('TN mean, std, ', tn_m.mean(), tn_m.std())

    fn_idxs = labels[threshold_index:] == 1 # Above threshold (i.e., labelled negative), but should be positive
    fn_m = ((m_a[idxs[threshold_index:]]) [fn_idxs])#.mean() 
    FN = np.sum(fn_idxs) # Above threshold (i.e., labelled negative), but should be positive
    print('FN mean, std, ', fn_m.mean(), fn_m.std())
    return float(FP) / float(FP + TN)

'''import operator


def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores.sort(key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count
'''
