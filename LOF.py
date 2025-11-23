#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:11:14 2024

@author: arda
"""
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOFAlgorithm:
    def LofProcess(self,df):
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        y_pred = clf.fit_predict(df)
        return df.join(pd.DataFrame(y_pred, columns=['outlier']),how='inner')
    