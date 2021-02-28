# # # # # # # # # # # # # # # #                   R-CNN                # # # # # # # # # # # # # # # # # # # #
# #  See r-cnn.svg for image illustration Selective search is performed on the input image to select multiple
# high-quality proposed regions. These proposed regions are generally selected on multiple
# scales and have different shapes and sizes. The category and ground-truth bounding box of each proposed region is
# labeled.
#
# A pre-trained CNN is selected and placed, in truncated form, before the output layer. It transforms each proposed
# region into the input dimensions required by the network and uses forward computation to output the features
# extracted from the proposed regions.
#
# The features and labeled category of each proposed region are combined as an example to train multiple support
# vector machines for object classification. Here, each support vector machine is used to determine whether an
# example belongs to a certain category.
#
# # The features and labeled bounding box of each proposed region are combined as an example to train a linear
# regression model for ground-truth bounding box prediction.

# # # # # # # # # # # # # # # # # #            Fast R-CNN                       # # # # # # # # # # # # # # # # # #
# See fast-rcnn.svg for image Compared to an R-CNN model, a Fast R-CNN model uses the entire image as the CNN input
# for feature extraction, rather than each proposed region. Moreover, this network is generally trained to update the
# model parameters. As the input is an entire image, the CNN output shape is  1×c×h1×w1 .
#
# Assuming selective search generates  n  proposed regions, their different shapes indicate regions of interests (
# RoIs) of different shapes on the CNN output. Features of the same shapes must be extracted from these RoIs (here we
# assume that the height is  h2  and the width is  w2 ). Fast R-CNN introduces RoI pooling, which uses the CNN output
# and RoIs as input to output a concatenation of the features extracted from each proposed region with the shape
# n×c×h2×w2 .
#
# A fully connected layer is used to transform the output shape to  n×d , where  d  is determined by the model design.
#
# During category prediction, the shape of the fully connected layer output is again transformed to  n×q  and we use
# softmax regression ( q  is the number of categories). During bounding box prediction, the shape of the fully
# connected layer output is again transformed to  n×4 . This means that we predict the category and bounding box for
# each proposed region.

# # # # # # # # # # # # # # # # # #            Faster R-CNN                       # # # # # # # # # # # # # # # # # #
# See faster-rcnn.svg for img
# The detailed region proposal network computation process is described below:
#
# We use a  3×3  convolutional layer with a padding of 1 to transform the CNN output and set the number of output
# channels to  c . This way, each element in the feature map the CNN extracts from the image is a new feature with a
# length of  c .
#
# We use each element in the feature map as a center to generate multiple anchor boxes of different sizes and aspect
# ratios and then label them.
#
# We use the features of the elements of length  c  at the center on the anchor boxes to predict the binary category
# (object or background) and bounding box for their respective anchor boxes.
#
# Then, we use non-maximum suppression to remove similar bounding box results that correspond to category predictions
# of “object”. Finally, we output the predicted bounding boxes as the proposed regions required by the RoI pooling
# layer.

# # # # # # # # # # # # # # # # # #            Mask R-CNN                       # # # # # # # # # # # # # # # # # #
# See mask-rcnn.svg for img


# # # # # # # # # # # # # # # # # # #           ROL-pooling                 # # # # # # # # # # # # # # # # # # #
# See roi.svg for image

from d2l import AllDeepLearning as d2l
from mxnet import nd
X = nd.arange(16).reshape((1, 1, 4, 4))
rois = nd.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
print(nd.ROIPooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1))


# # # # # # # # # # # # # # # #                    SUMMARY              # # # # # # # # # # # # # An R-CNN model
# selects several proposed regions and uses a CNN to perform forward computation and extract the features from each
# proposed region. It then uses these features to predict the categories and bounding boxes of proposed regions.
#
# Fast R-CNN improves on the R-CNN by only performing CNN forward computation on the image as a whole. It introduces
# an RoI pooling layer to extract features of the same shape from RoIs of different shapes.
#
# Faster R-CNN replaces the selective search used in Fast R-CNN with a region proposal network. This reduces the
# number of proposed regions generated, while ensuring precise object detection.
#
# Mask R-CNN uses the same basic structure as Faster R-CNN, but adds a fully convolution layer to help locate objects
# at the pixel level and further improve the precision of object detection.
