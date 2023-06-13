"""Data processing utilities."""

import json
import math
from texttable import Texttable
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged

def top_k(num_list, n):
    n %= len(num_list)
    pad = min(num_list)-1 
    topn_list = []
    index_list = []
    for i in range(n):
        topn_list.append(max(num_list))
        max_idx = num_list.index(max(num_list))
        index_list.append(max_idx)
        num_list[max_idx] = pad 
    return index_list

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """

    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation
    
def _calculate_prec_at_k(k, target):
    target_increase = np.sort(target)
    target_value_sel = (target_increase <= target_increase[k-1]).sum()
    if target_value_sel > k:
        best_k_target = target.argsort()[:target_value_sel]
    else:
        best_k_target = target.argsort()[:k]
    return best_k_target


def calculate_prec_at_k(k, prediction, target, target_ged):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = _calculate_prec_at_k(k, -target)
    best_k_target_ged = _calculate_prec_at_k(k, target_ged)

    
    return len(set(best_k_pred).intersection(set(best_k_target_ged))) / k 