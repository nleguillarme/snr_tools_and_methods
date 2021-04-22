import os
import codecs
import re
import pandas as pd
from copy import deepcopy

TAXON_LABELS = ["Taxon", "LIVB", "Microorganism", "Species"]


def get_labels(path_to_labels):
    """
    Returns a list of strings containing the annotations from file at
    path_to_labels with TX counter removed.
    Args:
            path_to_labels: path to annotation file in brat-standoff format
    Returns:
            dictionary of lists containing annotations for each file at path_to_labels
    """
    global_offsets = []
    global_labels = {}

    for file in os.listdir(path_to_labels):
        filename = os.fsdecode(file)
        if (not filename.startswith(".")) and (
            filename.endswith(".ann") or filename.endswith(".a2")
        ):
            with codecs.open(
                os.path.join(path_to_labels, filename), "r", encoding="utf-8"
            ) as test:
                labels = test.readlines()

            fname = filename.split(".")[0]
            # keep only offsets

            local_labels = []
            for x in labels:
                if re.split(r"[\t\s]", x.strip())[1] in TAXON_LABELS:
                    items = x.strip().replace(" ", "\t").split("\t")
                    offsets = items[2:4]
                    text = " ".join(items[4:])
                    local_labels += [
                        {"start": int(offsets[0]), "end": int(offsets[1]), "text": text}
                    ]

            # print(local_labels)

            global_labels[fname] = local_labels

    return global_labels


# Entity = {file, start, end, text}
def exact(pred, label):
    return pred["start"] == label["start"] and pred["end"] == label["end"]


def approximate(pred, label):
    return pred["start"] <= label["start"] and pred["end"] >= label["end"]


def left(pred, label):
    return pred["start"] == label["start"]


def right(entity1, entity2):
    return pred["end"] == label["end"]


def get_FN_FP_TP(predictions, labels, criterion):
    FPs = []
    TPs = []
    FNs = []

    for file in labels:

        label_annot = labels[file]
        # print(predictions)
        pred_annot = deepcopy(predictions[file])

        # TP
        for i in range(len(label_annot)):
            for j in range(len(pred_annot)):
                if criterion(pred_annot[j], label_annot[i]):
                    TPs.append({"gold": label_annot[i], "pred": pred_annot[j]})
                    pred_annot.pop(j)
                    break

        pred_annot = deepcopy(predictions[file])

        # FP
        for i in range(len(pred_annot)):
            match = False
            for j in range(len(label_annot)):
                if criterion(pred_annot[i], label_annot[j]):
                    match = True
                    break
            if not match:
                FPs.append(pred_annot[i])

        # FN
        for i in range(len(label_annot)):
            match = False
            for j in range(len(pred_annot)):
                if criterion(pred_annot[j], label_annot[i]):
                    match = True
                    break
            if not match:
                FNs.append(label_annot[i])

    return FNs, FPs, TPs


def get_FN_FP_TP_single_corpus(path_to_pred, path_to_gold, criterion=exact):
    """
    Returns sets of FN, FP and FP for predictions at path_to_pred
    for ground truth labels at path_to_gold using matching criterion.
    Args:
            path_to_pred: path to annotation predictions in brat-standoff format
            path_to_gold: path to annotation ground-truth in brat-standoff format
    criterion: matching criterion (one of {exact, approximate, left, right})
    Returns:
            3-tuple of FN, FP and TP for predictions based on ground truth.
    """
    # get predictions and labels
    predictions, labels = get_labels(path_to_pred), get_labels(path_to_gold)
    # get FP, FN, TP
    FN, FP, TP = get_FN_FP_TP(predictions, labels, criterion)

    return FN, FP, TP


def get_precision_recall_f1_single_corpus(path_to_pred, path_to_gold, criterion=exact):

    FN, FP, TP = get_FN_FP_TP_single_corpus(path_to_pred, path_to_gold, criterion)
    FN, FP, TP = len(FN), len(FP), len(TP)

    PRE = TP / (TP + FP)
    REC = TP / (TP + FN)
    F1 = 2 * PRE * REC / (PRE + REC)

    metrics = {"precision": [PRE], "recall": [REC], "f1-score": [F1]}
    df = pd.DataFrame(data=metrics, index=["Taxon"])
    print("${:.2f}$ & ${:.2f}$ & ${:.2f}$".format(PRE * 100, REC * 100, F1 * 100))
    return df
