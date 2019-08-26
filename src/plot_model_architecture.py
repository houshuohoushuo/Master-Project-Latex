from keras.utils import plot_model
from resnet_model import resnet18
from small_model import small
from vgg19_model import VGG19

input_shape = (200, 200, 1)
n_classes = 2

vgg = VGG19(input_shape, n_classes)
resnet = resnet18(input_shape, n_classes)
s = small(input_shape, n_classes)
plot_model(s, to_file='small_model.png', show_shapes=True)
plot_model(vgg, to_file='vgg_model.png', show_shapes=True)
plot_model(resnet, to_file='resnet_model.png', show_shapes=True)
