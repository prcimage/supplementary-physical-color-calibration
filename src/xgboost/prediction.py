#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import xgboost
import src.xgboost.features as features

def average_pred(pred):
    n = len(pred)
    if n == 1:
        print("Only one set of predictions, no ensemble averaging.")
    pr = pred[0].copy()
    for i in range(1, n):
            pr += pred[i]

    pred = pr / n
    return pred

def pred_to_class_vector(pred, thres_benign=None):
    """Classify ISUP predictions based on argmax (with possible exception for
    the benign class.

    Args:
        pred: (numpy array) 2d outputs from predict methods.
        thres_benign: if available, the threshold for positivity.

    Return:
        Argmax of preds, and if thres_benign then argmax over the grades.
    """
    if thres_benign:
        cl = pred[:, 1:].argmax(axis=1)
        cl = cl + 1
        cl = np.where(pred[:, 0] > thres_benign, 0, cl)
    else:
        cl = pred.argmax(axis=1)

    return cl

def standardize_rows(mat):
    s = mat.sum(axis=1)
    mat_std = mat / s[:, np.newaxis]
    return mat_std

def risk_by_cost(pred, lossmat):
    """Turn probabilities into risk scores by Bayes decision rule.

    We are interested in the conditional risk (expected loss given x),
    R(a | x) = y p(y | x)L(y, a), where L(y, a) defines the loss of action a
    when the truth is y. The optimal decision in this case is argmin a R(a | x).

    Arg:
        pred: a 1D array of predictions.
        lossmat: a (reasonable, but arbitrary) loss matrix rows are actions
            yhat, cols are true state y.

    Return: a 1D array of the same shape as pred.

    Cred to Mattias Rantalainen for providing this solution.
    """
    assert len(pred) == lossmat.shape[1], "dimensions not matching"
    # assert abs(sum(pred) - 1) < 1e-10, "pred not normalized"

    risk = lossmat.dot(pred)
    return risk


def risk_by_cost_wrap(pred, lossmat):
    risks = np.apply_along_axis(lambda x: risk_by_cost(x, lossmat), 1, pred)
    risks = np.apply_along_axis(lambda x: 1 / x, 1, risks)
    risks = standardize_rows(risks)
    return risks

def predict_wrap(df,
                 path_in_model,
                 path_out_predictions = None,
                 outcome = 'ISUP'):

    """Predict using slide-level classifier and tile predictions listed in a DataFrame.

    Args:
        df: DataFrame with tile level predictions for training.
        path_in_model: Path to file where trained model is stored.
        path_out_predictions: Output path for saving predictions on disk as .csv (default: None).
        outcome: Grading ('ISUP'), cancer detection ('cx') or length prediction ('CA_length') (default: 'ISUP').
    """

    assert outcome in ['ISUP', 'CA_length', 'cx'], 'No valid outcome!'
    
    # Load model(s).
    with open(path_in_model, "rb") as fp:
        model = pickle.load(fp)
    
    # Initialize features.    
    slidefeatures = features.default_features()

    # Collect feature values X and labels Y for all slides.
    if outcome == 'ISUP':
        # Get features for ISUP grading. The predictions for the first class
        # (benign) are kept but inverted to represent the probability of cancer
        # of any grade.
        X, Y = features.agg_features(df,
                                     features = slidefeatures,
                                     add_n_max = True,
                                     add_tile_count = True,
                                     skip_ben = False,
                                     inv_ben = True)
    else:
        # Get features for cancer detection.
        # The predictions for the first class (benign) are omitted, since they
        # are the complement probability of the second class (cancer).
        X, Y = features.agg_features(df,
                                     features = slidefeatures,
                                     add_n_max = True,
                                     add_tile_count = True,
                                     skip_ben = True,
                                     inv_ben = False)
    
    # Create Xgboost data set from X.
    dtest = xgboost.DMatrix(X)
    
    # Predict.
    pred = model.predict(dtest)
    
    # Put predictions into a DataFrame.
    if outcome == "ISUP":
        cols = ["slide_pred_isup_"+str(x) for x in range(6)]
    elif outcome == 'cx':
        cols = ["slide_pred_cancer"]
    elif outcome == 'CA_length':
        cols = ["slide_pred_mm"]
    df_out = pd.DataFrame(data=pred, columns=cols)

    # Add column indicating which slide each row corresponds to.
    df_out["slide"] = Y["slide"]
    
    # Output as csv.
    if not path_out_predictions is None:
        df_out.to_csv(path_out_predictions, index = False)
            
    return df_out