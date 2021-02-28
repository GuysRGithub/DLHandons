from mxnet import gluon, nd


"""
####################            BAYESIAN PERSONALIZED LOSS    ##################
    In formal, the training data is constructed by tuples in the form of  (u,i,j) ,
    which represents that the user  u  prefers the item  i  over the item  j .
    The Bayesian formulation of BPR which aims to maximize the posterior probability is given below:

    p(Θ∣>u)∝p(>u∣Θ)p(Θ)
    Where  Θ  represents the parameters of an arbitrary recommendation model,
        >u  represents the desired personalized total ranking of all items for user  u


    BPR-OPT:= =∑(u,i,j∈D)lnσ(y`ui−y`uj)−λΘ∥Θ∥^2
    where  D:={(u,i,j)∣i∈I+u∧j∈I∖I+u}  is the training set, with  I+u  denoting the items
    the user  u  liked,  I  denoting all items, and  I∖I+u  indicating all other items
    excluding items the user liked.  y^ui  and  y^uj  are the predicted scores of the
    user  u  to item  i  and  j , respectively. The prior  p(Θ)  is a normal distribution
    with zero mean and variance-covariance matrix  ΣΘ


"""


class BPRLoss(gluon.loss.Loss):
    def hybrid_forward(self, F, x, *args, **kwargs):
        pass

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = -nd.sum(nd.log(nd.sigmoid(distances)), 0, keepdims=True)
        return loss


class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None,
                                            batch_axis=batch_axis, **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = nd.sum(nd.maximum(- distances + margin, 0))
        return loss

    def hybrid_forward(self, F, x, *args, **kwargs):
        pass


