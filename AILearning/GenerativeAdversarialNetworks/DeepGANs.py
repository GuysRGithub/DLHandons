from d2l import AllDeepLearning as d2l
from mxnet import gluon, nd, init
from mxnet.gluon import nn
from AI.AILearning.GenerativeAdversarialNetworks import GANs as helper

data_dir = "E:/Python_Data/pokemon/pokemon"
pokemon = gluon.data.vision.datasets.ImageFolderDataset(data_dir)

batch_size = 256
transformer = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(64),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize(0.5, 0.5)])

data_iter = gluon.data.DataLoader(pokemon.transform_first(transformer),
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=0)

# d2l.set_figsize((4, 4))
# for X, y in data_iter:
#     imgs = X[0:20, :, :, :].transpose((0, 2, 3, 1), ) / 2 + 0.5
#     d2l.show_images(imgs, num_rows=4, num_cols=5)
#     break


# d2l.plt.show()
"""
    The generator needs to map the noise variable  z∈Rd , 
    a length- d  vector, to a RGB image with width and height to be  64×64
    
    The basic block of the generator contains a transposed convolution layer 
    followed by the batch normalization and ReLU activation.
"""


class G_block(nn.Block):
    def __init__(self, channels, kernel_size=4, strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.Conv2DTranspose(
            channels, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.Activation('relu')

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


"""
    In default, the transposed convolution layer uses a  kh=kw=4  kernel, 
    a  sh=sw=2  strides, and a  ph=pw=1  padding. With a input shape of  
    n′h×n′w=16×16 , the generator block will double input’s width and height.
    
    n′h×n′w
            =[(n_h * k_h−(n_h−1)(k_h−s_h)−2p_h]×[(n_w*k_w−(n_w−1)(k_w−s_w)−2p_w]
            =[(k_h+s_h(n_h−1)−2p_h]×[(k_w+s_w(n_w−1)−2p_w]
            =[(4+2×(16−1)−2×1]×[(4+2×(16−1)−2×1]
            =32×32.
"""

# x = nd.zeros((2, 3, 16, 16))
# g_blk = G_block(20)
# g_blk.initialize()

"""
    The generator consists of four basic blocks that increase input’s both 
    width and height from 1 to 32. At the same time, it first projects 
    the latent variable into  64×8  channels, and then halve the channels 
    each time. At last, a transposed convolution layer is used to generate 
    the output. It further doubles the width and height to match the desired  
    64×64  shape, and reduces the channel size to  3 . The tanh activation function
     is applied to project output values into the  (−1,1)  range.
"""

n_G = 64
net_G = nn.Sequential()
net_G.add(G_block(n_G * 8, strides=1, padding=0),  # output: (64*8, 4, 4)
          G_block(n_G * 4),  # output: (64*4, 8, 8)
          G_block(n_G * 2),  # output: (64*2, 16, 16)
          G_block(n_G),  # output: (64, 32, 32)
          nn.Conv2DTranspose(3, kernel_size=4, strides=2,
                             padding=1, use_bias=False, activation='tanh'))  # output: (3, 64, 64)

# x = nd.zeros((1, 100, 1, 1))
# net_G.initialize()
# print(net_G(x).shape)

"""
    leaky ReLU(x)={x if x>0, otherwise αx}
"""

# alphas = [0, 0.2, .4, .6, .8, 1]
# x = nd.arange(-2, 1, 0.1)
# Y = [nn.LeakyReLU(alpha)(x).asnumpy() for alpha in alphas]
# d2l.plot(x.asnumpy(), Y, 'x', 'y', alphas)
# d2l.plt.show()

"""

    The basic block of the discriminator is a convolution 
    layer followed by a batch normalization layer and a leaky ReLU
    activation. The hyper-parameters of the convolution layer are 
    similar to the transpose convolution layer in the generator block
    
"""


class D_block(nn.Block):
    def __init__(self, channels, kernel_size=4, strides=2,
                 padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2D(channels, kernel_size, strides,
                                padding, use_bias=False)
        self.batch_norm = nn.BatchNorm()
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


"""
        
    A basic block with default settings will halve the width and height of the inputs
    
    n′h×n′w
            =⌊(nh−kh+2ph+sh)/sh⌋×⌊(nw−kw+2pw+sw)/sw⌋
            =⌊(16−4+2×1+2)/2⌋×⌊(16−4+2×1+2)/2⌋
            =8×8.
        
"""

n_D = 64
net_D = nn.Sequential()
net_D.add(D_block(n_D),  # output: (64, 32, 32)
          D_block(n_D * 2),  # output: (64*2, 16, 16)
          D_block(n_D * 4),  # output: (64*4, 8, 8)
          D_block(n_D * 8),  # output: (64*8, 4, 4)
          nn.Conv2D(1, kernel_size=4, use_bias=False))  # output: (1, 1, 1)

"""
    It uses a convolution layer with output channel  1  
    as the last layer to obtain a single prediction value
"""

"""
    Compared to the basic GAN in Section 16.1, we use the same learning
    rate for both generator and discriminator since they are similar to 
    each other. In addition, we change  β1  in Adam (Section 11.10) from  
    0.9  to  0.5 . It decreases the smoothness of the momentum, the exponentially
    weighted moving average of past gradients, to take care of the rapid changing 
    gradients because the generator and the discriminator fight with each other.

"""


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim,
          ctx=d2l.try_gpu()):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True, ctx=ctx)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True, ctx=ctx)
    trainer_hp = {'learning_rate': lr, "beta1": 0.5}
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', trainer_hp)
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', trainer_hp)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs],
                            nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs + 1):

        timer = d2l.Timer()
        metric = d2l.Accumulator(3)
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = nd.random.normal(0, 1, shape=(batch_size, latent_dim, 1, 1))
            X, Z = X.as_in_context(ctx), Z.as_in_context(ctx)
            metric.add(helper.update_D(X, Z, net_D, net_G, loss, trainer_D),
                       helper.update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)

        # Show generated examples
        Z = nd.random.normal(0, 1, shape=(21, latent_dim, 1, 1), ctx=ctx)
        # Normalize the synthetic data to N(0, 1)
        fake_X = net_G(Z).transpose((0, 2, 3, 1)) / 2 + 0.5
        imgs = nd.concat(*[nd.concat(*[fake_X[i * 7 + j] for j in range(7)], dim=1)
                           for i in range(len(fake_X) // 7)], dim=0)
        animator.axes[1].cla()
        animator.axes[1].imshow(imgs.asnumpy())
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch, (loss_D, loss_G))
    print('loss_D % .3f, loss_G %.3f, %d examples/sec on %s' % (
        loss_D, loss_G, metric[2] / timer.stop(), ctx))


latent_dim, lr, num_epochs = 100, 0.005, 40
train(net_D, net_G, data_iter, num_epochs, lr, latent_dim)
d2l.plt.show()


"""
##########################3            SUMMARY             #############################
    DCGAN architecture has four convolutional layers for the Discriminator and four
     “fractionally-strided” convolutional layers for the Generator.

    The Discriminator is a 4-layer strided convolutions with batch normalization
    (except its input layer) and leaky ReLU activations.

    Leaky ReLU is a nonlinear function that give a non-zero output for
    a negative input. It aims to fix the “dying ReLU” problem and helps 
    the gradients flow easier through the architecture.
"""