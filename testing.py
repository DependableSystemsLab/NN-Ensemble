from util import (
    gpu_init,
    model_wrapper as mw,
    mnistmodel as mm,
    cifar10model as cm,
    gtsrbmodel as gb,
)


def test_model(benchmark):
    for modelx in model_list:
        mw.test(modelx)


mnist = [mm.AlexNet(), mm.CNN(), mm.LeNet(), mm.NN(), mm.ResNet50(), mm.RNN(), mm.VGG16()]
cifar10 = [cm.ConvNet(), cm.DeconvNet(), cm.MobileNet(), cm.ResNet18(), cm.ResNet50(), cm.VGG11(), cm.VGG16()]
gtsrb = [gb.AlexNet(), gb.CNN(), gb.LeNet(), gb.NN(), gb.ResNet50(), gb.RNN(), gb.VGG16()]


test_model(mnist)
test_model(cifar10)
test_model(gtsrb)

