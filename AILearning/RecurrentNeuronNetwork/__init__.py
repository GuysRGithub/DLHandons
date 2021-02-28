"""
    The idea of an RNN is a
    network with fixed input and output, which is being applied to the sequence
    of objects and can pass information along this sequence. This information is
    called hidden state and is normally just a vector of numbers of some size.
    On the following diagram, we have an RNN with one input which is a vector
    of numbers, the output of which is another vector. What makes it different
    from a standard feed-forward or convolution network is two extra gates: one
    input and one output. Extra input feeds the hidden state from the previous
    item into the RNN unit and the extra output provides a transformed hidden
    state to the next sequence.
"""