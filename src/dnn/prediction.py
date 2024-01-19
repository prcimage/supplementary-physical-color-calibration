#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import numpy as np
import pandas as pd
from skimage import io
import tensorflow as tf

tf.keras.mixed_precision.experimental.set_policy('float32')
tf.keras.backend.set_floatx('float32')
tf.keras.backend.set_image_data_format('channels_last')

class DNNPredictor:
    def __init__(self,
                 df,
                 path_in_model,
                 numcores=4,
                 batchsize=16):

        """Predict using a DNN.
        
        Args:
            df: DataFrame with tile data for testing.
            path_in_model: Path to trained model file.
            numcores: Number of CPU cores to use for preparing data (default: 4).
            queue: Number of batches per CPU core to keep queued in RAM (default: 2).
            batchsize: Number of tiles per minibatch per GPU (default: 16).
        """
        # Input environment parameters.
        self.df = df
        self.path_in_model = path_in_model
        self.numcores = int(numcores)
        self.batchsize = int(batchsize)

        # Internal variables.
        self.pred_names = None
        self.model = None
        
    def __find_classes(self):
        """Figure out the number of classes."""
        # Find number of output neurons i.e. number of classes.
        numclasses = self.model.output_shape[-1]
        
        # Define the name of the columns containing the class predictions.
        self.pred_names = ['tile_pred_class_' + str(i) for i in range(numclasses)]

    def init_model(self):
        """Load and compile model ready for prediction."""        
        
        # Load trained model.
        self.model = tf.keras.models.load_model(self.path_in_model)

    def predict(self):
        """Run predictions on the tiles listed in the DataFrame."""

        # Find out number of classes and output column names.
        self.__find_classes()
        
        # Tiles to predict.
        n_tot = len(self.df)
        
        # Number of full batches
        steps = n_tot // self.batchsize
        
        # Tiles that fit in full batches.
        n = steps * self.batchsize
        
        # Initialize Sequence for obtaining batches.
        pred_gen = PredictSequence(df = self.df,
                                   batch_size = self.batchsize)
        
        # Predict on all full batches.
        pred = self.model.predict(x = pred_gen,
                                  steps = steps,
                                  workers = self.numcores - 1,
                                  use_multiprocessing = False,
                                  max_queue_size = self.batchsize,
                                  verbose = 0)
        
        # Predict on the tiles that did not fit in full batches.
        df_left = self.df.iloc[n:n_tot]
        if not df_left.empty:
            pred_2 = []
            for tilename in df_left.tile_name:
                img = io.imread(tilename)
                
                # Prepare tile to match Keras input format.
                img = np.expand_dims(img, axis=0)
                img = np.vstack([img])
                img = np.array(img, np.float32) / 255
                
                # Run prediction and append to list.
                predtmp = self.model.predict(img).tolist()[0]
                pred_2.append(predtmp)
            
            # Combine full-batch and individual tile predictions.
            pred_2 = np.array(pred_2, dtype='float32')
            pred = np.vstack((pred, pred_2))
        
        # Add predictions to dataframe.
        df_pred = pd.DataFrame(pred, columns = self.pred_names)
        
        return df_pred

    def clear_model(self):
        """Clean up GPU memory and remove model."""
        tf.keras.backend.clear_session()
        del self.model
        self.pred_names = None
        self.model = None
        gc.collect()

class PredictSequence(tf.keras.utils.Sequence):
    """A sequence "generator". This multiprocessing-based
    method is faster than the multithreading-based generators."""
    def __init__(self,
                 df,
                 batch_size):
        
        self.tilename = df['tile_name']
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.tilename) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_tilename = self.tilename[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_tilename = batch_tilename.reset_index(drop=True)

        x_batch = []
        
        # Collect batch of tiles.
        for tilename in batch_tilename:
            img = io.imread(tilename)
            x_batch.append(img)

        x_batch = np.array(x_batch, np.float32) / 255

        return x_batch

def predict_wrap(df,
                 path_in_model,
                 path_out_predictions,
                 numcores = 4,
                 batchsize = 16):
    """Predict on tiles listed in a DataFrame using trained DNN.

    Args:
        df: DataFrame with tile data for testing.
        path_in_tiles: Path to folder where tiles are stored.
        path_in_model: Path to trained model file.
        path_out_predictions: Output file for saving predictions.
        numcores: Number of CPU cores to use for preparing data (default: 4).
        batchsize: Number of tiles per patch per GPU (default: 16).
    """
    
    # Initialize DNNPredictor.
    Predictor = DNNPredictor(df = df,
                             path_in_model = path_in_model,
                             numcores = numcores,
                             batchsize = batchsize)
    
    # Build model.
    Predictor.init_model()

    # Run prediction.
    df_pred = Predictor.predict()
    
    # Add prediction columns to DataFrame.
    df_pred = pd.concat([df, df_pred], axis=1)

    # Save DataFrame with predictions.
    df_pred.to_csv(path_out_predictions, index=False)

    # Clean up GPUs.
    Predictor.clear_model()
