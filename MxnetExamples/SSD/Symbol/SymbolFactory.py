import logging
from AI.MxnetExamples.SSD.Symbol import SymbolBuilder


def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks
        Parameters
        ----------
        network : str
            base network name, such as vgg_reduced, inceptionv3, resnet...
        data_shape : int
            input data dimension
        kwargs : dict
            extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[.07, .1025], [.15, .2121], [.3, .3674], [.45, .5196], [.6, .6708],
                     [.75, .8216], [.9, .9721]]
            ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3],
                      [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for
                                                  x in [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = [[.1, .141], [.2, .272], [.37, .447],
                     [.54, .619], [.71, .79], [.88, .961]]
            ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3],
                      [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warning("data_shape %d was not tested, use with cautious" % data_shape)
        return locals()
    elif network == "inceptionv3":
        from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat',
                       '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2, .272], [.37, .447],
                 [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3],
                  [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == "resnet50":
        num_layers = 50
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3],
                  [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
        normalizations = -1
        step = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3],
                  [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = "No configuration found ofr %s with data_shape %d" % (network, data_shape)
        raise NotImplementedError(msg)


def get_symbol_train(network, data_shape, **kwargs):
    """
        Wrapper for get symbol for train
        Parameters
        ----------
        network : str
            name for the base network symbol
        data_shape : int
            input shape
        kwargs : dict
            see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warning("Using legacy model.")
        return SymbolBuilder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return SymbolBuilder.get_symbol_train(**config)


def get_symbol(network, data_shape, **kwargs):
    """
        Wrapper for get symbol for test
        Parameters
        ----------
        network : str
            name for the base network symbol
        data_shape : int
            input shape
        kwargs : **kwargs (dict)
    """
    if network.startswith('legacy'):
        logging.warning("Using legacy model.")
        return SymbolBuilder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return SymbolBuilder.get_symbol_train(**config)



