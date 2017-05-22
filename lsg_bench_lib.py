import lasagne
import theano
import theano.tensor as T
import numpy as np
from timeit import default_timer as timer


def toy_batch(seed, shape):
    np.random.seed(seed)
    bX = np.float32(np.random.uniform(-1,1,(shape)))
    bY = [0]
    b_lenX= [shape[1]]
    return bX, bY, b_lenX

def get_train_and_val_fn(input_var, mask_var, target_var, network, lr):
    # Get final output of network
    prediction = lasagne.layers.get_output(network)
    # Calculate the loss with categorical cross entropy
    loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=target_var)
    loss = loss.mean()

    # Acquire all the parameters recursively in the network
    params = lasagne.layers.get_all_params(network, trainable=True)

    # Use default adam learning
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # Get a deterministic output for test-time, in case we use dropout
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Get the loss according to the deterministic test-time output
    test_loss = lasagne.objectives.categorical_crossentropy(predictions=prediction, targets=target_var)
    test_loss = test_loss.mean()

    # Group all the inputs together
    fn_inputs = [input_var, mask_var, target_var]
    # Compile the training function
    train_fn = theano.function(fn_inputs, loss, updates=updates)
    # Compile the test function
    val_fn = theano.function(fn_inputs, test_loss)
    # compile the prediction function
    pred_fn = theano.function([input_var, mask_var], test_prediction)
    return train_fn, val_fn, pred_fn

def get_bench_net(input_var, mask_var, inp_dim, rnn_size, out_size, GRAD_CLIP):
    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, None, inp_dim), input_var=input_var)

    # Masking layer
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var)

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    # RNN layers
    h1 = lasagne.layers.GRULayer(l_in, num_units=rnn_size, mask_input=l_mask, grad_clipping=GRAD_CLIP,
                                  hid_init=lasagne.init.GlorotUniform())
    h2 = lasagne.layers.SliceLayer(h1, -1, axis=1)
    h3 = lasagne.layers.DenseLayer(h2, num_units=out_size, nonlinearity=lasagne.nonlinearities.softmax)

    return h3

def lsg_rnn(rnn_size=200, out_size =11, batch_shape = (1,85,39), batch_seed = 11, GRAD_CLIP = 200, lr=1e3 ):
    inp_dims = batch_shape[2]
    # Create symbolic vars
    input_var = T.ftensor3('my_input_var')
    mask_var = T.matrix('my_mask')
    target_var = T.ivector('my_targets')

       # Get network
    network = get_bench_net(input_var, mask_var, inp_dims, rnn_size, out_size, GRAD_CLIP)

    print('# network parameters: ' + str(lasagne.layers.count_params(network)))

    # Compile
    train_fn, val_fn, pred_fn = get_train_and_val_fn(input_var, mask_var, target_var, network, lr)

    # Training

    time = []

    for ep in range(100):

        train_loss = 0
        cnt = 0
        bX, bY, b_lenX = toy_batch(batch_seed, batch_shape)
        maskX = np.float32(np.ones((1,batch_shape[1])))

        start = timer()
        loss = train_fn(bX, maskX, bY)
        end = timer()
        time.append(end - start)

        cnt += 1
        train_loss += loss

    return (np.mean(time), np.min(time), np.max(time), len(time),lasagne.layers.count_params(network))