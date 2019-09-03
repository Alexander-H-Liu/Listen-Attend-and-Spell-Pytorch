# post_process_scores.py
"""
    compute AUC and other metrics on scores.
"""
import yaml
import torch
import random
import argparse
import numpy as np
import os
import logging
import pandas as ps
from sklearn import svm
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support
import scipy.stats
import sklearn.metrics

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='post processing.')
    parser.add_argument('--scores', type=str, help='scores')
    parser.add_argument('--split', type=str, help='split')
    parser.add_argument('--input_data', type=str, help='input_data') # forreading labels
    paras = parser.parse_args()

    inps = ps.read_csv(paras.input_data)
    fnames, labels = inps["file_path"], inps["label"]
    files2label = {}
    for fn, lab in zip(fnames, labels):
        files2label[fn] = lab

    
    data = ps.read_csv(paras.scores)
    splits, fnames, scores = data["split"], data["fname"], data["score"]
    Y = []
    X = []
    for sp, fn, sc in zip(splits, fnames, scores):
        if sp == paras.split:
            lab = files2label[fn]
        
            Y.append(int(lab))
            X.append(float(sc))

    ##print(Y)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, X, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)

    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    tpr_eer = 1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    specificity, sensitivity = 1-fpr_eer, tpr_eer
    y_pred_bin = [float(y >= eer_threshold)  for y in list(X)]
    cm = confusion_matrix(Y, y_pred_bin)
    total = 1.0 * sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision, recall, f1, support = precision_recall_fscore_support(Y, y_pred_bin)
    print("AUC: ", auc)
    print("EER specificity, sensitivity:", specificity, sensitivity)
    print("EER cm:", cm)
    print("EER accuracy:", accuracy)
    print("EER precision, recall, f1, support:", precision, recall, f1, support)

if __name__ == "__main__":
    main()
