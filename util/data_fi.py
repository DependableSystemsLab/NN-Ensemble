from src import tfi, config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses

import csv
import json
import numpy as np
import sys
import time
import math

def write_to_csv(
    modelx,
    fi_type,
    fi_amt,
    filter_ids,
    test_labels,
    results,
    analysis_name,
    timestamp,
    csv_header):

    if (fi_type == "baseline"):
        file_name = "./analysis/" + modelx.get_name() + "-" + fi_type + str(fi_amt) + "-" + \
                                                analysis_name +".csv"
    else:
        file_name = "./analysis/" + modelx.get_name() + "-" + fi_type + str(fi_amt) + "-" + \
                                                analysis_name + "-" + timestamp +".csv"

    if not (len(filter_ids) == len(test_labels) == len(results)):
        print("Filter ids, test labels and results are different lengths!")
        return

    with open(file_name, "w") as w_file:
        csvwriter = csv.writer(w_file, delimiter=',',
                               quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(csv_header)

        for (a, b, c) in zip(filter_ids, test_labels, results):
            csvwriter.writerow([a, b, c])

    print("Writing complete:", file_name)


def train_data_generator(train_images, train_labels, capacity=15000):
    length = len(train_images)
    lower = 0
    for lower in range(0, length, capacity):
        upper = lower + capacity
        yield train_images[lower:upper], train_labels[lower:upper]
        lower = lower + capacity

def perform_data_fi(
    modelx,
    conf_file,
    fi_type,
    train_images,
    train_labels):

    img_slices = None
    lab_slices = None

    for tr_img_slice, tr_lab_slice in train_data_generator(train_images, train_labels):

        if (fi_type == "label_err"):
            tf_res = tfi.inject(y_test=tr_lab_slice, confFile=conf_file)

            if lab_slices is None:
                lab_slices = tf_res
            else:
                lab_slices = np.concatenate((lab_slices, tf_res), axis=0)

        else:
            tf_res = tfi.inject(x_test=tr_img_slice,
                                y_test=tr_lab_slice, confFile=conf_file)

            if img_slices is None:
                img_slices = tf_res[0]
            else:
                img_slices = np.concatenate((img_slices, tf_res[0]), axis=0)

            if lab_slices is None:
                lab_slices = tf_res[1]
            else:
                lab_slices = np.concatenate((lab_slices, tf_res[1]), axis=0)

    if (fi_type == "label_err"):
        test_loss, test_acc = modelx.train_fi(train_images, lab_slices)
    else:
        test_loss, test_acc = modelx.train_fi(img_slices, lab_slices)

    return modelx, test_acc


def prune_golden(modelx, golden_labels):
    test_images, test_labels = modelx.get_test_data()
    pruned_test_images = [test_images[i] for i in golden_labels]
    pruned_test_labels = [test_labels[i] for i in golden_labels]
    pruned_test_images = np.asarray(pruned_test_images)
    pruned_test_labels = np.asarray(pruned_test_labels)

    return golden_labels, pruned_test_images, pruned_test_labels


def prune_single_golden(modelx):
    with open("./golden_labels/" + modelx.get_name() + "-golden-labels.txt", "r") as r_file:
        golden_labels = json.load(r_file)
        return prune_golden(modelx, golden_labels)


def prune_multiple_golden(modelx):
    with open("./golden_labels/unified-golden-labels.txt", "r") as r_file:
        golden_labels = json.load(r_file)
        return prune_golden(modelx, golden_labels)


def measure_sdc(modelx, pruned_test_images, pruned_test_labels, verbose=True):
    test_loss, test_acc = modelx.evaluate(pruned_test_images, pruned_test_labels)
    sdc = 1 - test_acc
    if verbose:
        print("SDC Rate:", sdc)
    return sdc


def save_predictions(modelx, fi_type, fi_amt, test_images, test_labels):
    filter_ids = [i for i in range(0, len(test_labels))]
    csv_header = ["id", "test_label", modelx.get_plain_name()]
    timestamp = str(math.floor(time.time()))

    predicted_list = modelx.predict(test_images)
    predictions = np.argmax(predicted_list, axis = 1)
    test_labels = test_labels.flatten()
    write_to_csv(modelx, fi_type, fi_amt, filter_ids, test_labels, predictions, "predictions", timestamp, csv_header)

    corrected_labelled = np.equal(predictions, test_labels)
    corrected_labelled = corrected_labelled.astype(int)
    write_to_csv(modelx, fi_type, fi_amt, filter_ids, test_labels, corrected_labelled, "correct_pred", timestamp, csv_header)


def get_stats_checkpoint_filename(modelx, fi_type, fi_amt):
    return "./analysis/acc_sdc_stats/" + modelx.get_name() + "-" + fi_type + str(fi_amt) + "-" + \
                                                "stats_checkpoint" + ".csv"

def checkpoint_results(file_name, acc, sdc):
    with open(file_name, "a") as a_file:
        csvwriter = csv.writer(a_file, delimiter=',',
                               quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([acc, sdc])


def fi_model(modelx, conf_file="./confFiles/sample.yaml"):
    fi_conf = config.config(conf_file)
    fi_type = fi_conf["Type"]
    fi_amt = fi_conf["Amount"]
    stats_checkpoint_filename = get_stats_checkpoint_filename(modelx, fi_type, fi_amt)

    # Default Verbosity
    verbose = False
    modelx.set_verbose(0)

    (train_images, train_labels), (test_images, test_labels) = modelx.get_data()
    golden_labels, pruned_test_images, pruned_test_labels = prune_single_golden(modelx)


    modelx, test_acc = perform_data_fi(modelx, conf_file, fi_type, train_images, train_labels)

    sdc = measure_sdc(modelx, pruned_test_images, pruned_test_labels, verbose)

    checkpoint_results(stats_checkpoint_filename, test_acc, sdc)
    save_predictions(modelx, fi_type, fi_amt, test_images, test_labels)
    modelx.clear_weights()

    print(stats_checkpoint_filename)


def create_baseline(modelx):
    fi_type = "baseline"
    fi_amt = 0

    test_images, test_labels = modelx.get_test_data()
    modelx.checkpoint_load()
    save_predictions(modelx, fi_type, fi_amt, test_images, test_labels)

