# Introduction

Large transformer based language models have given rise to open-ended language generation. Likes of ChatGpt, LLama, T5 and Gemma are some of the examples of such models.
There are three pieces of good language models:

1. **Training Data**: Massive unsupervised data is used to train the model.
2. **Model Architecture**: Improved transformer based architecture.
3. **Decoding Strategy**: Better decoding strategy to generate the text.

Greedy search and beam search are the Deterministic decoding strategies.
Sampling, Top-k sampling and Top-p sampling are stochastic decoding strategies.

Generally, it is challenging to find a prompt and model-independent temperature that avoids both the pitfalls of greedy search and nucleus sampling.

## Prominent Decoding Strategies

### Greedy Search

It is the simplest [decoding strategy](https://huggingface.co/blog/how-to-generate#greedy-search) . It selects the token with the highest probability at each time step. It is fast but not very accurate. <br/>
$w_t = argmax_w P(w|w_{t:t-1})$

The generated words following the context is reasonable, but the overall coherence of the generated text is not good. This problem occurs in greedy and beam search.
The major drawback of greedy search though is that it misses high probability words hidden behind a low probability word.


### Beam Search

[It](https://huggingface.co/blog/how-to-generate#beam-search) reduces the risk of missing high probability words by keeping most likely **num_beams** hypothesis at each time step and eventually choosing the hypothesis that has the overall probability.
It is a heuristic search algorithm that explores a graph by expanding the most promising node in a limited set. It will always find an output with higher probability than greedy search, but it is still not guaranteed to find the most likely output.

The output still include repetitions of same words sequences. To mitigate this, **n-gram** penalties are introduces. The n-gram penalty is a penalty for generating the same n-gram multiple times.
The most common n-grams penalty makes sure that no n-gram appears twice by setting the probability of next words that could create an already seed n-gram to 0.
This penalty should be used with care as where repetition is not a problem, it can hurt the quality of the output.

The reason why beam search is not best possible option:

1. It works well in tasks where the length of the generation is more ore less predictable as in machine translation or summarization. For task where output length can vary like dialog or story generation, it is not the best option.
2. It suffers from repetetive genration. It is hard to control the repetition of words in the output as finding the correct trade-off between inhibiting repetition and allowing for natural repetition of words is difficult.
3. As high quality human language does not follow a distribution of high probability words, as human look for surprise in the text not the most probable word as its boring/predictable.

### Sampling

[Sampling](https://huggingface.co/blog/how-to-generate#sampling) means model randomly selects the next word according to its conditional probability distribution. 

$ w_t \text{\textasciitilde} P(w|w_{1:t-1}) $

The text generated using sampling is not deterministic. It is not guaranteed to be coherent and thats a problem when sampling word sequence.
To make distribution sharper by increasing the likelihood of high probability words, we can use temperature parameter. It is a hyperparameter that controls the randomness of the output. Lower temperature means less randomness and higher temperature means more randomness.
This temperature is associated with the softmax function.

#### Top-k Sampling

In [Top-K Sampling](https://huggingface.co/blog/how-to-generate#top-k-sampling), K most likely words are selected and the probability mass is redistributed among them. GPT2 adopted this sampling strategy. The text from this strategy is more coherent and most human sounding text. One concern with this strategy is that it doens not adapt the number of wrods that are filtered from the next word probability distribution.
AS some words can be sampled from a very sharp distribution and some from a very flat distribution. Limitng the sample pool to a fixed size K could endager the model to produce givverish for shapr distribution and limit the diversity of the output for flat distribution.

#### Top-p Sampling

[Top-p sampling](https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling) chosses from the smallest possible set of words whose cumulative probability exceeds the probability p. The probability mass is then redistributed among this set of words.
This strategy is more flexible than top-k sampling as it adapts the number of words that are filtered from the next word probability distribution. It is also known as nucleus sampling.
Top-p can also be combined with top-k to avoid very low ranked words while allowing for dynamic selection.

### [Contrastive Search](https://huggingface.co/blog/introducing-csearch)

Given the prefix text $x_{<t}$, the selection of the output token $x_t$ follows:

$ x_t = argmax_{\substack{v \epsilon V^{(k)}}} \lbrace (1- \alpha) * p_{\theta}(v|x_{<t}) - \alpha * (max{
    s(h_v, h_{xj}: 1 \leq j \leq t-1)
})\rbrace $

where $V^{(k)}$ is the set of k most likely tokens from the language model's probability distribution $p_{\theta}(v|x_{<t})$.
The first term is model confidence $p_{\theta}(v| x_{<t})$, is the probability of the candidate v predicted by the language model. Second term, _degeneration penalty_, measures how discriminative of v with respect to the previous context $x_{<t} $ and the function $s(h_v, h_{xj})$ is a cosine similarity function between the token representation $h_v$ and the prefix token representation $h_{xj}$.
Larger degenration penalty means that the token is more similar to the previous tokens and hence less likely to be selected. As most similar tokens leds to the problem of model degeneration. When $\alpha = 0$, the model is equivalent to greedy search and when $\alpha = 1$, the model is equivalent to sampling.

_When generating output, contrastive search jointly considers (i) the probability predicted by the language model to maintain the semantic coherence between the generated text and the prefix text; and (ii) the similarity with respect to the previous context to avoid model degeneration._

## References

[Decoding Strategies](https://huggingface.co/docs/transformers/generation_strategies#decoding-strategies)
[Contrastive Search](https://huggingface.co/blog/introducing-csearch)
[Constrastive Search explain](https://github.com/yxuansu/SimCTG/blob/main/contrastive_search_explanation/README.md)