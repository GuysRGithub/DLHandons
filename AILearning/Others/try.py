from mxnet import nd, contrib, image
from d2l import AllDeepLearning as d2l
img = image.imread('E:\\Python_Data\\dog_cat.jpg').asnumpy()
d2l.plt.imshow(img)
d2l.plt.show()

