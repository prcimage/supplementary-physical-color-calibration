#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
import numpy as np
import pandas as pd

def default_features():
    features = [np.sum, np.median, np.max, percentile_9975, percentile_995,
                percentile_98, percentile_9925, percentile_99, percentile_95,
                percentile_90, percentile_80, percentile_10, hist_999, hist_99,
                hist_90]
    return features

def percentile_9975(x):
    return np.percentile(x, 99.75)
def percentile_995(x):
    return np.percentile(x, 99.5)
def percentile_9925(x):
    return np.percentile(x, 99.25)
def percentile_99(x):
    return np.percentile(x, 99)
def percentile_98(x):
    return np.percentile(x, 98)
def percentile_95(x):
    return np.percentile(x, 95)
def percentile_90(x):
    return np.percentile(x, 90)
def percentile_80(x):
    return np.percentile(x, 80)
def percentile_10(x):
    return np.percentile(x, 10)
def percentile_5(x):
    return np.percentile(x, 5)
def percentile_1(x):
    return np.percentile(x, 1)
def percentile_05(x):
    return np.percentile(x, 0.5)

def hist_999(x):
    return x.gt(.999).sum().astype(int)
def hist_99(x):
    return x.gt(.99).sum().astype(int)
def hist_90(x):
    return x.gt(.9).sum().astype(int)

def agg_features(df,
                 features,
                 add_n_max = True,
                 add_tile_count = True,
                 skip_ben = False,
                 inv_ben = True):

    """Aggregate histogram-type features by slide from tile predictions.

    Args:
        df: Pandas DataFrame with tile predictions.
        features: (list) numpy accknowledged functions of numeric vectors.
        add_n_max: (bool) add n argmax for each class
        add_tile_count: (bool) include tiles per slide as feature.
        skip_ben: (bool) do not include predictioins for benign.
        inv_ben: (bool) switch pred benign to pred cancer.
    """
    assert not (skip_ben and inv_ben), "Cannot both skip and invert!"

    # Get names of columns representing tile-level predictions.
    name_pred = df.columns[df.columns.str.startswith('tile_pred_')].tolist()

    # Drop first column corresponding to the benign class if required.
    if skip_ben:
        name_pred = name_pred[1:]
    
    # Grouping all tiles from a slide together.
    dg = df.groupby(['slide'])

    # For each slide, calculate the number of tiles for each class having the
    # highest probability over all classes. This produces a slides x classes
    # table of tile counts.
    if add_n_max:
        df['amax'] = df[name_pred].idxmax(1)
        amax = dg['amax'].value_counts().sort_index()
        amax = pd.DataFrame({'n': amax})
        amax.reset_index(inplace=True)

        amax = pd.pivot_table(amax,
                              values = 'n',
                              index = ['slide'],
                              columns = ['amax'],
                              aggfunc = np.max)

        # If some classes have no tiles at all, insert zeros.
        amax = amax.fillna(0)
        amax = amax.astype('int')

        # Check that all class columns are present, add zeros if needed.
        want = set(name_pred)
        has = set(name_pred).intersection(set(amax.columns))
        if want > has:
            add = list(want - has)
            for col in add:
                amax[col] = 0
            amax = amax.reindex(columns=name_pred)
    
    # If required, invert the probability of the benign class to instead
    # represent the probability of cancer.
    if inv_ben:
        df[name_pred[0]] = 1 - df[name_pred[0]]

    # Create an empty DataFrame with the slide as index.
    dfs = [dg.agg({'slide': 'first'}).drop(['slide'],axis=1)]

    # Add DataFrame representing the number of tiles per slide to the list.
    if add_tile_count:
        dg0 = dg[name_pred[0]].agg(['count'])
        dfs.append(dg0)
    
    # Aggregate all features in class-wise manner over all tiles for each slide.
    # Append resulting DataFrame to the list.
    for i in range(len(name_pred)):
        dg0 = dg[name_pred[i]].agg(features)
        ending = i
        if skip_ben:
            ending += 1
        dg0.columns = dg0.columns + '_' + str(ending)
        dfs.append(dg0)

    # Join tables on index (slide, split and ISUP, the latter to keep outcome).
    df_final = reduce(lambda left, right: pd.merge(left,
                                                   right,
                                                   left_index=True,
                                                   right_index=True), dfs)
    
    # Add number of tiles per class having the highest class-wise probability.
    if add_n_max:
        df_final = pd.concat([df_final, amax], axis=1, join='inner')

    df_final = df_final.reset_index()
    
    y_cols = ['slide']

    x = df_final.drop(y_cols, axis=1)
    y = df_final[y_cols]    

    return x, y
