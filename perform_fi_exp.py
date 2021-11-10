#! /usr/bin/python3

import argparse
from util import (
    gpu_init,
    mnistmodel as mm,
    cifar10model as cm,
    gtsrbmodel as gt,
    data_fi as dfi,
)


parser = argparse.ArgumentParser()

parser.add_argument('-b', action='store', dest='benchmark',
                    help='Benchmark name: mnist, cifar10')

parser.add_argument('-m', action='store', dest='modelname',
                    help='Example model names: alexnet, cnn, lenet, nn, resnet50, rnn, vgg16')

parser.add_argument('--silent', action='store_true', default=False,
                    dest='silent',
                    help='Run silently, default=False')

def alexnet_mnist():
    dfi.fi_model(mm.AlexNet())


def cnn_mnist():
    dfi.fi_model(mm.CNN())


def lenet_mnist():
    dfi.fi_model(mm.LeNet())


def nn_mnist():
    dfi.fi_model(mm.NN())


def resnet50_mnist():
    dfi.fi_model(mm.ResNet50())


def rnn_mnist():
    dfi.fi_model(mm.RNN())


def vgg16_mnist():
    dfi.fi_model(mm.VGG16())


def convnet_cifar10():
    dfi.fi_model(cm.ConvNet())


def deconvnet_cifar10():
    dfi.fi_model(cm.DeconvNet())


def mobilenet_cifar10():
    dfi.fi_model(cm.MobileNet())


def resnet18_cifar10():
    dfi.fi_model(cm.ResNet18())


def resnet50_cifar10():
    dfi.fi_model(cm.ResNet50())


def vgg3_cifar10():
    dfi.fi_model(cm.VGG3())


def vgg16_cifar10():
    dfi.fi_model(cm.VGG16())


def alexnet_gtsrb():
    dfi.fi_model(gt.AlexNet())


def cnn_gtsrb():
    dfi.fi_model(gt.CNN())


def lenet_gtsrb():
    dfi.fi_model(gt.LeNet())


def nn_gtsrb():
    dfi.fi_model(gt.NN())


def resnet50_gtsrb():
    dfi.fi_model(gt.ResNet50())


def rnn_gtsrb():
    dfi.fi_model(gt.RNN())


def vgg16_gtsrb():
    dfi.fi_model(gt.VGG16())


def main():
    results = parser.parse_args()

    benchmark = results.benchmark
    modelname = results.modelname
    silent = results.silent

    if not silent:
        print("Model chosen is", modelname, "and will be injected faults", "time(s) on", benchmark, "\n")
    globals()[modelname + '_' + benchmark]()


if __name__ == "__main__":
    main()

