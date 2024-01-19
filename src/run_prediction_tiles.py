#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import glob
import os
import pandas as pd
import src.dnn.prediction as dnnprediction

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_in_tiles", help="Path to folder with tiles", type=str, default="example_data/original/slide_1")
    parser.add_argument("--path_out_tilepredictions", help="Output path for saving predictions", type=str, default="example_results/original/slide_1")
    parser.add_argument("--path_in_model_cancer", help="Path to folder with trained models for cancer detection", type=str, default="models/original/tile_level/cancer")
    parser.add_argument("--path_in_model_grading", help="Path to folder with trained models for grading", type=str, default="models/original/tile_level/grading")
    parser.add_argument("--numcores", help="Number of parallel processes", type=int, default=4)
    parser.add_argument("--batchsize", help="Number of tiles per patch per GPU", type=int, default=16)
    P = parser.parse_args()
    
    # Create output folder.
    if not os.path.exists(P.path_out_tilepredictions):
        os.makedirs(P.path_out_tilepredictions)
    
    # Read the DataFrame with project tile data.
    df = pd.DataFrame(glob.glob(os.path.join(P.path_in_tiles,"*")), columns=['tile_name'])
    df['slide'] = P.path_in_tiles
    
    # Run predictions with all cancer detection models.
    paths_in_model_cancer = sorted(glob.glob(os.path.join(P.path_in_model_cancer,"*.tf")))
    
    for i, path_in_model in enumerate(paths_in_model_cancer):
        print("Predicting on cancer detection model "+str(i+1)+'/'+str(len(paths_in_model_cancer))+": "+path_in_model)
        dnnprediction.predict_wrap(df = df,
                                   path_in_model = path_in_model,
                                   path_out_predictions = os.path.join(P.path_out_tilepredictions,'predictions_tiles_cancer_'+str(i)+'.csv'),
                                   numcores = P.numcores,
                                   batchsize = P.batchsize)
        
    # Run predictions with all grading models.
    paths_in_model_grading = sorted(glob.glob(os.path.join(P.path_in_model_grading,"*.tf")))
    
    for i, path_in_model in enumerate(paths_in_model_grading):
        print("Predicting on grading model "+str(i+1)+'/'+str(len(paths_in_model_cancer))+": "+path_in_model)
        dnnprediction.predict_wrap(df = df,
                                   path_in_model = path_in_model,
                                   path_out_predictions = os.path.join(P.path_out_tilepredictions,'predictions_tiles_grading_'+str(i)+'.csv'),
                                   numcores = P.numcores,
                                   batchsize = P.batchsize)
