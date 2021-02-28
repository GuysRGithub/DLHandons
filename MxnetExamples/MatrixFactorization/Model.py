import mxnet as mx


def matrix_fact_model_parallel_net(factor_size, num_hidden, max_user,
                                   max_item):
    with mx.AttrScope(ctx_group='dev1'):
        user = mx.symbol.Variable('user')
        item = mx.symbol.Variable("item")

        user_weight = mx.symbol.Variable("user_weight")
        user = mx.symbol.Embedding(data=user, weight=user_weight,
                                   input_dim=max_user, output_dim=factor_size)
        item_weight = mx.symbol.Variable('item_weight')
        item = mx.symbol.Embedding(data=item, weight=item_weight,
                                   input_dim=max_item, output_dim=factor_size)

    with mx.AttrScope(ctx_group='dev2'):
        user = mx.symbol.Activation(data=user, act_type='relu')
        fc_user_weight = mx.symbol.Variable('fc_user_weight')
        fc_user_bias = mx.symbol.Variable("fc_user_bias")
        user = mx.symbol.FullyConnected(data=user, weight=fc_user_weight,
                                        bias=fc_user_bias, num_hidden=num_hidden)

        item = mx.symbol.Activation(data=item, act_type='relu')
        fc_item_weight = mx.symbol.Variable('fc_item_weight')
        fc_item_bias = mx.symbol.Variable('fc_item_bias')
        item = mx.symbol.FullyConnected(data=item, weight=fc_item_weight,
                                        bias=fc_item_bias, num_hidden=num_hidden)

        pred = user * item
        pred = mx.symbol.sum(data=pred, axis=1)
        pred = mx.symbol.Flatten(data=pred)
        score = mx.symbol.Variable('score')

        pred = mx.symbol.LinearRegressionOutput(data=pred, label=score)
    return pred
