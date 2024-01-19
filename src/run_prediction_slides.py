#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import glob
import os
import numpy as np
import pandas as pd
import src.xgboost.prediction as xgboostprediction

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_in_tilepredictions", help="Path to the folder with tile-level prediction CSV files", type=str, default="example_results/original/slide_1")
    parser.add_argument("--path_out_slidepredictions", help="Output path to CSV file for saving slide-level predictions", type=str, default="example_results/original/slide_1/predictions_slides.csv")
    parser.add_argument("--path_in_model_cancer", help="Path to trained cancer detection model file", type=str, default="models/original/slide_level/xgbmodel_cancer.pkl")
    parser.add_argument("--path_in_model_grading", help="Path to trained grading model file", type=str, default="models/original/slide_level/xgbmodel_grading.pkl")
    parser.add_argument("--path_in_model_length", help="Path to trained cancer length model file", type=str, default="models/original/slide_level/xgbmodel_length.pkl")
    parser.add_argument("--threshold_cancer", help="Minimum probability for assigning a slide as malignant with original model, the value for calibrated model is 0.423473.", type=float, default=0.955624)
    P = parser.parse_args()
        
    # Create output folder.
    if not os.path.exists(os.path.dirname(P.path_out_slidepredictions)):
        os.makedirs(os.path.dirname(P.path_out_slidepredictions))
    
    # Get all the dataframes with tile-level predicton
    path_in_dataframe_cancer = sorted(glob.glob(os.path.join(P.path_in_tilepredictions,"*cancer*.csv")))
    path_in_dataframe_grading = sorted(glob.glob(os.path.join(P.path_in_tilepredictions,"*grading*.csv")))
    
    # List of slides to predict on.
    slides = []
    
    # ISUP grading.
    # Loop through tile-level predictions from DNN models for grading, and run
    # the slide-level prediction using each of them as input.
    predictions_grading = []
    
    for ensembledf in path_in_dataframe_grading:
        # Read the DataFrame with tile-level predictions.
        df = pd.read_csv(ensembledf)
        
        # Run predictions for ISUP grading.
        pred = xgboostprediction.predict_wrap(df,
                                              path_in_model = P.path_in_model_grading,
                                              path_out_predictions = None,
                                              outcome = 'ISUP')
        slides.append(pred['slide'])
        pred = np.array(pred.drop(columns=['slide']))
        predictions_grading.append(pred)
    
    # Average predictions over the ensemble.
    predictions_grading = xgboostprediction.average_pred(predictions_grading)
    
    # Apply Bayesian weighting to decision rule.
    p = 2
    lossmat = np.array([[0, p * .1, p * .2, p * .3, p * .4, p * .5],
                        [.1, 0, p * .1, p * .2, p * .3, p * .4],
                        [.2, .1, 0, p * .1, p * .2, p * .3],
                        [.3, .2, .1, 0, p * .1, p * .2],
                        [.4, .3, .2, .1, 0, p * .1],
                        [.5, .4, .3, .2, .1, 0]])
    predictions_grading = xgboostprediction.risk_by_cost_wrap(predictions_grading,
                                                              lossmat)
    
    # Cancer detection.
    # Loop through tile-level predictions from DNN models for cancer detection,
    # and run the slide-level prediction using each of them as input.
    predictions_cancer = []
    for ensembledf in path_in_dataframe_cancer:
        # Read the DataFrame with tile-level predictions.
        df = pd.read_csv(ensembledf)
        
        # Run predictions for cancer detection.
        pred = xgboostprediction.predict_wrap(df,
                                              path_in_model = P.path_in_model_cancer,
                                              path_out_predictions = None,
                                              outcome = 'cx')
        slides.append(pred['slide'])
        pred = np.array(pred.drop(columns=['slide']))
        predictions_cancer.append(pred)              
        
    # Average predictions over the ensemble.
    predictions_cancer = xgboostprediction.average_pred(predictions_cancer)

    # Cancer length estimation.
    # Loop through tile-level predictions from DNN models for cancer detection,
    # and run the slide-level prediction using each of them as input.
    predictions_length = []
    for ensembledf in path_in_dataframe_cancer:
        # Read the DataFrame with tile-level predictions.
        df = pd.read_csv(ensembledf)
        
        # Run predictions for cancer length.
        pred = xgboostprediction.predict_wrap(df,
                                              path_in_model = P.path_in_model_length,
                                              path_out_predictions = None,
                                              outcome = 'CA_length')
        slides.append(pred['slide'])
        pred = np.array(pred.drop(columns=['slide']))
        predictions_length.append(pred)
        
    # Average predictions over the ensemble.
    predictions_length = xgboostprediction.average_pred(predictions_length)

    # Replace negative length values with zeros.
    predictions_length = np.where(predictions_length < 0, 0, predictions_length)

    # Make sure all predictions were run on the same set of slides.
    for i in range(len(slides)):
        assert(all(slides[i] == slides[0])), "Mismatching slides across tile-level DataFrames!"
    # Keep only one set, since they are identical.
    slides = slides[0]
    
    # Incorporate cancer detection results with ISUP grading results by
    # replacing the probability of the benign class estimated by the ISUP
    # grading model with that of the cancer detection model.
    predictions_grading[:, 0] = 1 - np.squeeze(predictions_cancer)
    
    # Get ISUP classifications.
    class_isup = xgboostprediction.pred_to_class_vector(predictions_grading,
                                                        thres_benign = 1 - P.threshold_cancer)

    # Collect slide-level output into a DataFrame.
    df_out = pd.DataFrame(data = slides, columns = ['slide'])
    
    # Add ISUP grades as probabilities and as hard classification outcomes.
    df = pd.DataFrame(data = predictions_grading,
                      columns = ["slide_pred_isup_"+str(x) for x in range(6)])
    df_out = pd.concat([df_out, df], axis=1)
    df = pd.DataFrame(data = class_isup, columns = ["slide_class_isup"])
    df_out = pd.concat([df_out, df], axis=1)
    
    # Add cancer detection probability.
    df = pd.DataFrame(data = predictions_cancer, columns = ["slide_pred_cancer"])
    df_out = pd.concat([df_out, df], axis=1)
    
    # Add length estimates.
    df = pd.DataFrame(data = predictions_length, columns = ["slide_pred_length"])
    df_out = pd.concat([df_out, df], axis=1)

    # Output as CSV.
    df_out.to_csv(P.path_out_slidepredictions, index = False)
    
    
