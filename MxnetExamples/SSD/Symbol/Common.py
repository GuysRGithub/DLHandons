import mxnet as mx
import numpy as np


def conv_act_layer(from_layer, name, num_filter, kernel=(1, 1), pad=(0, 0),
                   stride=(1, 1), act_type="relu", use_batch_norm=False):
    """
        wrapper for a small Convolution group
        Parameters:
        ----------
        from_layer : mx.symbol
            continue on which layer
        name : str
            base name of the new layers
        num_filter : int
            how many filters to use in Convolution layer
        kernel : tuple (int, int)
            kernel size (h, w)
        pad : tuple (int, int)
            padding size (h, w)
        stride : tuple (int, int)
            stride size (h, w)
        act_type : str
            activation type, can be relu...
        use_batchnorm : bool
            whether to use batch normalization
        Returns:
        ----------
        (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad,
                                 stride=stride, num_filter=num_filter, name='{}_conv'.format(name))
    if use_batch_norm:
        conv = mx.symbol.BatchNorm(conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type,
                                name="{}_{}".format(name, act_type))
    return relu


def legacy_conv_act_layer(from_layer, name, num_filter, kernel=(1, 1), pad=(0, 0),
                          stride=(1, 1), act_type="relu", use_batch_norm=False):
    """
        wrapper for a small Convolution group
        Parameters:
        ----------
        from_layer : mx.symbol
            continue on which layer
        name : str
            base name of the new layers
        num_filter : int
            how many filters to use in Convolution layer
        kernel : tuple (int, int)
            kernel size (h, w)
        pad : tuple (int, int)
            padding size (h, w)
        stride : tuple (int, int)
            stride size (h, w)
        act_type : str
            activation type, can be relu...
        use_batchnorm : bool
            whether to use batch normalization
        Returns:
        ----------
        (conv, relu) mx.Symbols
    """
    assert not use_batch_norm, "batch norm not yet supported"
    bias = mx.symbol.Variable(name="conv{}_bias".format(name),
                              init=mx.init.Constant(0.0), attr={'__lr_mult__': 2.0})
    conv = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=kernel, pad=pad,
                                 stride=stride, num_filter=num_filter, name="conv{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, name="{}{}".format(act_type, name))
    if use_batch_norm:
        relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
    return conv, relu


def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
        layers and SSD specific layers
        Parameters
        ----------
        body:
        from_layers : list of str
            feature extraction layers, use '' for add extra layers
            For example:
            from_layers = ['relu4_3', 'fc7', '', '', '', '']
            which means extract feature from relu4_3 and fc7, adding 4 extra layers
            on top of fc7
        num_filters : list of int
            number of filters for extra layers, you can use -1 for extracted features,
            however, if normalization and scale is applied, the number of filter for
            that layer must be provided.
            For example:
            num_filters = [512, -1, 512, 256, 256, 256]
        strides : list of int
            strides for the 3x3 convolution appended, -1 can be used for extracted
            feature layers
        pads : list of int
            paddings for the 3x3 convolution, -1 can be used for extracted layers
        min_filter : int
            minimum number of filters used in 1x1 convolution
        Returns
        -------
        list of mx.Symbols
    """

    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    internals = body.get_internals()
    layers = []

    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % k,
                                      num_1x1, kernel=(1, 1), pad=(0, 0),
                                      stride=(1, 1), act_type='relu')
            conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % k,
                                      num_filter, kernel=(3, 3), pad=(p, p),
                                      stride=(s, s), act_type='relu')
            layers.append(conv_3x3)
    return layers


def multibox_layer(from_layers, num_classes, sizes=None,
                   ratios=None, normalization=-1, num_channels=None,
                   clip=False, interm_layer=0, steps=None):
    if ratios is None:
        ratios = [1]
    if num_channels is None:
        num_channels = []
    if steps is None:
        steps = []
    if sizes is None:
        sizes = [.2, .95]
    """
        the basic aggregation module for SSD detection. Takes in multiple layers,
        generate multiple object detection targets by customized layers
        Parameters:
        ----------
        from_layers : list of mx.symbol
            generate multibox detection from layers
        num_classes : int
            number of classes excluding background, will automatically handle
            background in this function
        sizes : list or list of list
            [min_size, max_size] for all layers or [[], [], []...] for specific layers
        ratios : list or list of list
            [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
        normalizations : int or list of int
            use normalizations value for all layers or [...] for specific layers,
            -1 indicate no normalizations and scales
        num_channels : list of int
            number of input layer channels, used when normalization is enabled, the
            length of list should equals to number of normalization layers
        clip : bool
            whether to clip out-of-image boxes
        interm_layer : int
            if > 0, will add a intermediate Convolution layer
        steps : list
            specify steps for each MultiBoxPrior layer, leave empty, it will calculate
            according to layer dimensions
        Returns:
        ----------
        list of outputs, as [loc_preds, cls_preds, anchor_boxes]
        loc_preds : localization regression prediction
        cls_preds : classification prediction
        anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, "num_classes {} must be larger than 0".format(num_classes)
    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must be same length"
    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        assert 0 < sizes[0] < 1
        assert sizes[0] < sizes[1] < 1
        tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers) - 1))
        start_offset = .1
        min_sizes = [start_offset] + list(tmp)
        max_sizes = list(tmp) + [tmp[-1] + start_offset]
        sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must be same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)

    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"
    if steps:
        assert len(steps) == len(from_layers), \
            "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, mode='channel',
                                                   name="%s_norm" % from_name)
            scale = mx.symbol.Variable(name="%s_scale" % from_name,
                                       shape=(1, num_channels.pop(0), 1, 1),
                                       init=mx.init.Constant(normalization[k]),
                                       attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3, 3),
                                               stride=(1, 1), pad=(1, 1),
                                               num_filter=interm_layer,
                                               name="%s_inter_conv" % from_name)
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu",
                                              name="%s_inter_relu" % from_name)

        # estimate number of anchors per location
        # here I follow the original version in caffe
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratios]) + ")"
        num_anchors = len(size) - 1 + len(ratio)

        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="%s_loc_pred_conv_bias" % from_name,
                                  init=mx.init.Constant(0.0),
                                  attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3, 3),
                                         stride=(1, 1), pad=(1, 1), num_filter=num_loc_pred,
                                         name="%s_loc_pred_conv" % from_name)
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0, 2, 3, 1))
        loc_pred = mx.symbol.Flatten(loc_pred)
        loc_pred_layers.append(loc_pred)

        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="%s_cls_pred_conv_bias" % from_name,
                                  init=mx.init.Constant(0.0),
                                  attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3, 3),
                                         stride=(1, 1), pad=(1, 1), num_filter=num_cls_pred,
                                         name="%s_cls_pred_conv" % from_name)
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0, 2, 3, 1))
        cls_pred = mx.symbol.Flatten(cls_pred)
        cls_pred_layers.append(cls_pred)

        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.symbol.contrib.MultiBoxPrior(from_layer, sizes=size_str,
                                                  ratios=ratio_str, clip=clip,
                                                  steps=step,
                                                  name="%s_anchors" % from_name)
        anchors = mx.symbol.Flatten(anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.concat(*loc_pred_layers, num_args=len(loc_pred_layers),
                                 dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.concat(*cls_pred_layers, num_args=len(cls_pred_layers),
                                 dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1),
                                    name="multibox_cls_pred")
    anchor_boxes = mx.symbol.concat(*anchor_layers, num_args=len(anchor_layers),
                                    dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4),
                                     name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]

