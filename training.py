from util import (
    gpu_init,
    model_wrapper as mw,
    mnistmodel as mm,
    cifar10model as cm,
    gtsrbmodel as gb,
)

mw.train(mm.AlexNet())
mw.train(mm.CNN())
mw.train(mm.LeNet())
mw.train(mm.NN())
mw.train(mm.ResNet50())
mw.train(mm.RNN())
mw.train(mm.VGG16())

mw.train(cm.ConvNet())
mw.train(cm.DeconvNet())
mw.train(cm.MobileNet())
mw.train(cm.ResNet18())
mw.train(cm.ResNet50())
mw.train(cm.VGG3())
mw.train(cm.VGG16())

mw.train(gb.AlexNet())
mw.train(gb.CNN())
mw.train(gb.LeNet())
mw.train(gb.NN())
mw.train(gb.ResNet50())
mw.train(gb.RNN())
mw.train(gb.VGG16())

