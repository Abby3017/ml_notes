# EdiT5: Semi-Autoregressive Text Editing with T5 Warm-Start

[Its](https://arxiv.org/pdf/2205.12209.pdf) a semi autoregressive text-editing model. It decompose generation process into three sub-tasks: (1) tagging to decide on the subset of input tokens to be preserved in the output, (2) re-ordering to define their order in the output text,
and (3) insertion to infill the missing tokens that are not present in the input.
The tagging and re-ordering steps, which are responsible for generating the largest portion of the output, are non-autoregressive, while the insertion step uses an autoregressive decoder.
EDIT5 is faster during inference than conventional sequence-to-sequence (seq2seq) models, while being capable of modeling flexible input-output transformations.
It is initialized with a pre-trained T5 checkpoint yielding comparable performanceto T5 in high-resource settings.

## Flexible text editing

EDIT5 supports open-vocabulary generation by relying on an autoregressive decoder.
In the extreme case, where there is no overlap between the source and the target texts, it reduces to a vanilla seq2seq model generating the entire output from scratch.
However, when the input and output overlap, it can benefit from the tagging and pointer networks to reconstruct the bulk of the output text that is further infilled (refined) by the autoregressive decoder.

## Warm Start

Training a high-precision text generation model typically requires large amounts of high-quality supervised data. Self-supervised techniques based on text in-filling (Rothe et al., 2020a; Lewis et al., 2020b; Raffel et al., 2020) have been shown to provide a crucial advantage over non-pre-trained models especially in low-resource settings.
Hence, we design EDIT5 to be able to benefit from already existing pre-trained language models (specifically T5), where the final model is directly fine-tuned on the downstream task.

## Model Description

The model architecture of EDIT5 resembles a vanilla Transformer composed of an encoder and a decoder. EDIT5 decomposes the generation of a text y from an input x into three parts:

1. predicting a sequence of edit tags $y^t$ (indicating whether a token from the input should be copied to the output). Modeled by Encoder.
2. a permutation of the input tokens $\pi$ (indicating the order that copied tokens should appear in in the output). Modeled by Encoder.
3. a sequence of tokens $y^d$ (indicating additional tokens that should be in the output, and where in the permuted input they should be inserted). Modeled by Decoder.

There are multiple ways to choose the triple $(y^t,\pi, y^d)$ for a given (x, y) pair. During dataset creation we choose a single such triple for each training pair. Probability of y is defined as:

$ P(y|x) = \lparen \prod_{i}^{|y^d|} P(y^d_i | y^d_{<i},y^T,\pi,,x) \rparen * P(\pi|y^T, x) * P(y^T|x) $ <br/>

For Inference, authors maximised probability of each term. For example, $y^T$ greedily set to maximize third term, then $\pi$ is maximised to second term and finally $y^d$ is maximised to first term.

### Text Editing Encoder

#### Encoder

The source sentence x is first encoded using N transformer layers into the hidden representations h.

#### Tagging

For tagging, token KEEP and DELETE is used to indicate whether a token from the input should be copied to the output or not. Tags are predicted by applying a single transformer layer followed by a classification layer to the output of the encoder h, which is trained using cross-entropy.

$L_{tagging} = - \Sigma_j^{|x|} log P(y_j^t|f_t(h)_j) $

where $y^t$ are the gold tags, j is the index of the source token, and $f_t$ is a transformer layer followed by a classification layer.
During inference we use argmax to determine the tags, whereas during training we use the gold tags.
The encoder hidden state is then updated to take these tags into account:

$ h^t_j = f_te([h_j;TE(y_j^t)]) $ <br/>

Where TE is a tag embedding layer, whose output is concatenated to the original hidden representation of the source sequence, before a feed-forward layer fte is applied.
