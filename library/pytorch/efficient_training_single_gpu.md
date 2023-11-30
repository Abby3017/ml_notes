# Methods and tools for efficient training on a single GPU

In this, we are going to look into techniques to increase the efficiency of your model’s training by optimizing memory utilization, speeding up the training, or both.

When training large models, there are two aspects that should be considered at the same time:

- Data throughput/training time
- Model performance

Maximizing the throughput (samples/second) leads to lower training cost. This is generally achieved by utilizing the GPU as much as possible and thus filling GPU memory to its limit. If the desired batch size exceeds the limits of the GPU memory, the memory optimization techniques, such as gradient accumulation, can help.

However, if the **preferred batch size fits into memory, there’s no reason to apply memory-optimizing techniques because they can slow down the training.** Just because one can use a large batch size, does not necessarily mean they should. As part of hyperparameter tuning, you should determine which batch size yields the best results and then optimize resources accordingly.

## Techniques

### Batch size Choice

To achieve optimal performance, start by identifying the appropriate batch size. It is recommended to use batch sizes and input/output neuron counts that are of size 2^N.
Often it’s a multiple of 8, but it can be higher depending on the hardware being used and the model’s dtype.

