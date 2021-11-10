from util import (
    gpu_init,
    model_wrapper as mw,
    mnistmodel as mm,
    cifar10model as cm,
    cifar100model as ch,
    gtsrbmodel as gb,
)


def test_model(benchmark):
    for modelx in model_list:
        mw.test(modelx)


mnist = [mm.AlexNet(), mm.CNN(), mm.LeNet(), mm.NN(), mm.ResNet50(), mm.RNN(), mm.VGG16()]
cifar10 = [cm.AlexNet(), cm.CNN(), cm.LeNet(), cm.NN(), cm.ResNet50(), cm.RNN(), cm.VGG16()]
cifar100 = [ch.AlexNet(), ch.CNN(), ch.LeNet(), ch.NN(), ch.ResNet50(), ch.RNN(), ch.VGG16()]
gtsrb = [gb.AlexNet(), gb.CNN(), gb.LeNet(), gb.NN(), gb.ResNet50(), gb.RNN(), gb.VGG16()]


test_model(mnist)
test_model(cifar10)
test_model(cifar100)
test_model(gtsrb)

