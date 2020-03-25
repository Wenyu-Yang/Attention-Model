from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM, RepeatVector, Dense, Activation
from keras.models import Model
from nmt_utils import softmax

repeator = RepeatVector(30)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)
post_activation_LSTM_cell = LSTM(64, return_state=True)
output_layer = Dense(11, activation=softmax)
    
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    
    s_prev = repeator(s_prev)
    
    concat = concatenator([a, s_prev])
    
    e = densor1(concat)
    
    energies = densor2(e)
    
    alphas = activator(energies)
    
    context = dotor([alphas, a])
    
    return context

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    for t in range(Ty):
    
        context = one_step_attention(a, s)
        
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
        
        out = output_layer(inputs=s)
        
        outputs.append(out)
    
    model = Model(inputs=[X, s0, c0], outputs=outputs)
    
    return model

