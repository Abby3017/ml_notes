# Gated Recurrent Unit (GRU)

GRUs are able to effectively retain long-term dependencies in sequential data. And additionally, they can address the “short-term memory” issue plaguing vanilla RNNs.
It is a variant of the RNN architecture, and uses gating mechanisms to control and manage the flow of information between cells in the neural network.
GRUs are a gating mechanism in recurrent neural networks, introduced in 2014 by Kyunghyun Cho et al. in the paper titled “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation”.
Other than its internal gating mechanisms, the GRU functions just like an RNN, where sequential input data is consumed by the GRU cell at each time step along with the memory, or otherwise known as the hidden state.
The hidden state is then re-fed into the RNN cell together with the next input data in the sequence. This process continues like a relay system, producing the desired output.

## GRU Architecture

The GRU architecture is similar to the LSTM architecture, but has fewer parameters than LSTM, as it lacks an output gate. GRU has two gates, a reset gate and update gate.
The **reset gate** determines how to combine the new input with the previous memory, and the **update gate** defines how much of the previous memory to keep.The gates are designed to protect and control the cell state.

<img src="https://blog.floydhub.com/content/images/2019/07/image14.jpg" alt="gru_internal" width="600" height="400"/>

### Reset Gate

This gate is derived and calculated using both the hidden state from the previous time step and the input data at the current time step.
This is achieved by multiplying the previous hidden state and current input with their respective weights and summing them before passing the sum through a sigmoid function.
The sigmoid function will transform the values to fall between 0 and 1, allowing the gate to filter between the less-important and more-important information in the subsequent steps. <br/>
$\text{gate}_{\text{reset}} = \sigma \lparen W_{\text{input\_reset}} \sdot x_{t} + W_{\text{hidden\_reset}} \sdot h_{t-1} \rparen $

The previous hidden state will first be multiplied by a trainable weight and will then undergo an element-wise multiplication (Hadamard product) with the reset vector. This operation will decide which information is to be kept from the previous time steps together with the new inputs.
At the same time, the current input will also be multiplied by a trainable weight before being summed with the product of the reset vector and previous hidden state above. Lastly, a non-linear activation tanh function will be applied to the final result to obtain r in the equation below.
<br/>
$\text{r} = \tanh \lparen \text{gate}_{\text{reset}} \odot \lparen W_{\text{h1}} \sdot h_{t-1} \rparen + \lparen W_{x1} \sdot x_{t} \rparen \rparen $

### Update Gate

The update gate is similar to the reset gate, but instead of deciding which information to keep from the previous time steps, it decides which information to keep from the current time step.
Both the Update and Reset gate vectors are created using the same formula, but, the weights multiplied with the input and hidden state are unique to each gate, which means that  the final vectors for each gate are different. This allows the gates to serve their specific purposes.
<br/>
$\text{gate}_{\text{update}} = \sigma \lparen W_{\text{input\_update}} \sdot x_{t} + W_{\text{hidden\_update}} \sdot h_{t-1} \rparen $

The Update vector will then undergo element-wise multiplication with the previous hidden state to obtain u in our equation below, which will be used to compute our final output later.
<br/>
$\text{u} = \text{gate}_{\text{update}} \odot h_{t-1} $

### Pytorch initialization of GRU Hidden State



### Final Output

The final output is computed by multiplying the update vector with the previous hidden state and the reset vector with the candidate vector.
This time, we will be taking the element-wise inverse version of the same Update vector (1 - Update gate) and doing an element-wise multiplication with our output from the Reset gate, r.
The purpose of this operation is for the Update gate to determine what portion of the new information should be stored in the hidden state.
<br/>
$\text{h}_{t} = \text{r} \odot \lparen 1 - \text{gate}_{\text{update}} \rparen + \text{u} $

### Solution to Vanishing Gradient Problem

The gates in the GRUs help to solve this problem because of the additive component of the Update gates.
While traditional RNNs always replace the entire content of the hidden state at each time step, LSTMs and GRUs keep most of the existing hidden state while adding new content on top of it.
This allows the error gradients to be back-propagated without vanishing or exploding too quickly due to the addition operations.

## GRU vs LSTM

While both GRUs and LSTMs contain gates, the main difference between these two structures lies in the number of gates and their specific roles. The role of the Update gate in the GRU is very similar to the Input and Forget gates in the LSTM.
However, the control of new memory content added to the network differs between these two.

<img src="https://blog.floydhub.com/content/images/2019/07/image12.jpg" alt="gru_vs_lstm_internal"/>

In the LSTM, while the Forget gate determines which part of the previous cell state to retain, the Input gate determines the amount of new memory to be added.
These two gates in the LSTM keep independent information while in GRU update gate is responsible for retention of previous memory and addition of new information.
GRU hold long term and short term memory in a single hidden state while LSTM has two separate states.
