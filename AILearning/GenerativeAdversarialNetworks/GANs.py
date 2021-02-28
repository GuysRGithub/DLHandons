from d2l import AllDeepLearning as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

X = nd.random.normal(shape=(1000, 2))
A = nd.array([[1, 2], [-.1, .5]])
b = nd.array([1, 2])
data = nd.dot(X, A) + b

batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
net_G = nn.Sequential()
net_G.add(nn.Dense(2))

net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))


def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = nd.ones((batch_size,), ctx=X.context)
    zeros = nd.zeros((batch_size,), ctx=X.context)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for net_G, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return loss_D.sum().asscalar()


def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = nd.ones((batch_size,), ctx=Z.context)
    with autograd.record():
        # Recomputing fake_Y is needed since net_D is changed.
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return loss_G.sum().asscalar()


def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', {"learning_rate": lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', {"learning_rate": lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2,
                            figsize=(5, 5), legend=['generator', 'discriminator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for X in data_iter:
            batch_size = X.shape[0]
            Z = nd.random.normal(0, 1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        Z = nd.random.normal(0, 1, shape=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real, generated'])
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print('loss_D %.3f, loss_G %.3f, %d examples/sec' % (loss_D, loss_G, metric[2] / timer.stop()))


# lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
# train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
#       latent_dim, data[:100].asnumpy())
# d2l.plt.show()
"""
   ##################3                     SUMMARY        ################
   Generative adversarial networks (GANs) composes of two deep networks, 
   the generator and the discriminator.

    The generator generates the image as much closer to the true image as 
    possible to fool the discriminator, via maximizing the cross-entropy loss,
     i.e.,  maxlog(D(x′)) .

    The discriminator tries to distinguish the generated images
     from the true images, via minimizing the cross-entropy loss, 
     i.e.,  min−ylogD(x)−(1−y)log(1−D(x)) .
"""
