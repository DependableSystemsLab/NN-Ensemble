import json
import numpy as np
import pandas as pd


def save_golden_labels(modelx):
    modelx.checkpoint_load()

    test_images, test_labels = modelx.get_test_data()
    test_labels = test_labels.flatten()
    predicted_list = modelx.predict()
    predictions = np.argmax(predicted_list, axis=1)
    comp = np.equal(predictions, test_labels)
    pruned_test_labels = [i for i in range(0, len(comp)) if comp[i]]

    print("Number of pruned test labels:", len(pruned_test_labels))

    with open("./golden_labels/" + modelx.get_name() + "-golden-labels.txt", "w") as w_file:
        json.dump(pruned_test_labels, w_file)


def save_golden_predictions(modelx):
    modelx.checkpoint_load()

    test_images, test_labels = modelx.get_test_data()
    test_labels = test_labels.flatten()
    predicted_list = modelx.predict()
    predictions = np.argmax(predicted_list, axis=1)

    pred_list = predictions.tolist()

    print("Predictions printed: ", len(pred_list))

    with open("./golden_pred/" + modelx.get_name() + "-golden.txt", "w") as w_file:
        json.dump(pred_list, w_file)


def train(modelx):
    modelx.train()
    save_golden_predictions(modelx)


def test(modelx):
    modelx.checkpoint_load()

    test_loss, test_acc = modelx.test()
    print("Accuracy before faults for", modelx.get_name(), "is:", test_acc)

