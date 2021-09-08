import json
import logging
from datetime import timedelta
from random import shuffle

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class JsonIfy(BaseEstimator, TransformerMixin):

    def __init__(self, inference=False, pred_length=None, dynamic_feat=True):
        self.inference = inference
        self.pred_length = pred_length
        self.dynamic_feat = dynamic_feat

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        time_series_jslines = []
        for ts in X:
            time_series_jslines.append(json.loads(series_to_jsonline(ts, dynamic_feat=self.dynamic_feat,
                                                                     pred_length=self.pred_length,
                                                                     inference=self.inference)))
        return time_series_jslines
    
def series_to_jsonline(ts, dynamic_feat=None, cat=None, inference=False, pred_length=None):
    if inference == False:
        return json.dumps(series_to_obj_train(ts, dynamic_feat, cat))
    else:
        return json.dumps(series_to_obj_inference(ts, dynamic_feat, cat, pred_length))
    

def series_to_obj_inference(ts, dynamic_feat=None, cat=None, pred_length=None):
    target = list(ts['target'])
    obj = {"start": str(ts.index[0]), "target": target[:-pred_length]}
    if cat is not None:
        obj["cat"] = cat

    if dynamic_feat is not None:
        dyn_feat_list = (list(ts['dynamic_features']))
        obj["dynamic_feat"] = [dyn_feat_list]

    return obj
