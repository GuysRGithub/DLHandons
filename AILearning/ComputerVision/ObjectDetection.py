from d2l import AllDeepLearning as d2l
from mxnet import image, nd, gluon, contrib

d2l.set_figsize((3.5, 2.5))
img = image.imread('E:\\Python_Data\\dog_cat.jpg').asnumpy()
h, w = img.shape[0:2]
print(h, w)
X = nd.random.uniform(shape=(1, 3, h, w))
Y = nd.contrib.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])

boxes = Y.reshape(h, w, 5, 4)
# dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# d2l.set_figsize((3.5, 2.5))
bbox_scale = nd.array((w, h, w, h))
# fig = d2l.plt.imshow(img)
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#             ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
#              's=0.75, r=0.5'])
# d2l.plt.show()


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])
# print(ground_truth.shape)
# fig = d2l.plt.imshow(img)
# show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale,
            # ['dog', 'cat'], 'k')
# show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
# d2l.plt.show()

labels = nd.contrib.MultiBoxTarget(nd.expand_dims(anchors, axis=0),
                                   nd.expand_dims(ground_truth, axis=0),
                                   nd.zeros((1, 3, 5)))
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,  # Predicted probability for background
                      [0.9, 0.8, 0.7, 0.1],  # Predicted probability for dog
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted probability for cat

fig = d2l.plt.imshow(img)
# show_bboxes(fig.axes, anchors * bbox_scale,
#             ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
output = nd.contrib.MultiBoxDetection(nd.expand_dims(cls_probs, axis=0),
                                      nd.expand_dims(offset_preds, axis=0),
                                      nd.expand_dims(anchors, axis=0),
                                      nms_threshold=0.5)

for i in output[0].asnumpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)

# print(output)
d2l.plt.show()
# fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
# fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
