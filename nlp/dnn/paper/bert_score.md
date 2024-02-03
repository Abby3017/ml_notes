# BERTScore

[BertScore](https://openreview.net/pdf?id=SkeHuCVFDr) computes a similarity score for each token in the candidate sentence with each token in the reference sentence.
The final score is the F1 score of the token similarity scores, weighted by the token length ratio. Token similarity is computed using contextual embedding.

## Definition

BLEU ,the most common machine translation metric, simply counts n-gram overlap between the candidate and the reference.
BERTSCORE computes the similarity of two sentences as a sum of cosine similarities between their tokens’ embeddings.

## Prior Metrics for generation evaluation metrics

n-Gram Matching approaches: BLEU, ROUGE, METEOR, CIDEr

Edit Distance approaches:
TER normalizes edit distance by the number of reference words.
ITER adds stem matching and better normalization.
PER computes position independent error rate
CDER models block reordering as an edit operation.

Word Embedding approaches: Embedding Average, Greedy Matching, Vector Extrema, Word Mover’s Distance
MEANT 2.0 uses word embeddings and shallow semantic parses to compute lexical and structural similarity.

Learned Embedding:
BEER uses a regression model based on character n-grams and word bigrams.
BLEND uses regression to combine 29 existing metrics.
RUSE combines three pre-trained sentence embedding models.

## How BERTScore works

BERT can generate different vector representations for the same word in different sentences depending on the surrounding words, which form the context of the target word.

1. First, Reference sentence $ x = \langle x_1, x_2,...., x_k \rangle $ and candidate sentence $ x' = \langle x'_1, x'_2,....,x'_k \rangle $ is tokenised.
2. Then, the tokenised sentences are passed through a pre-trained BERT model to get contextual embeddings for each token.
3. Similarity score is calculated between reference token $x_i$ and candidate token $x'_j$ as cosine similarity between their embeddings. As pre-noramlised vector is used, cosine similarity is equivalent to inner product $x_i^T \cdot x'_j$.
4. The complete score matches each token in x to a token in xˆ to compute recall, and each token in xˆ to a token in x to compute precision. To maximize similairty score, greedy matching is used, where each token is matched to the most similar token in the other sentence.
Precision and recall are combined to compute an F1 measure. <br/>
$ R_{BERT} = \frac{1}{|x|} \Sigma_{x_i \epsilon x} \text{max}_{x'_j \epsilon x'} x_i^T x'_j $ <br/>
$ P_{BERT} = \frac{1}{|x'|} \Sigma_{x'_j \epsilon x'} \text{max}_{x_i \epsilon x} x_i^T x'_j $
5. Importance weighting in BERTScore is based on idf logic.
6. Baseline Rescaling is done to make the score more readable. The constant b is calculated by averaging 1 M candidate-reference pairs from the random generation.
