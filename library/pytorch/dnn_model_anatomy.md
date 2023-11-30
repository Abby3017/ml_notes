# DNN model Anatomy

Training large models on GPUS efficiently requires know-how of operation and optimization available.

## Model Training Anatomy

File required to better analysis.

`pip install transformers datasets accelerate nvidia-ml-py3`

The nvidia-ml-py3 library allows us to monitor the memory usage of the models from within Python. 

For testing we create some dummy data: random token IDs between 100 and 30000 and binary labels for a classifier.
In total, we get 512 sequences each with length 512 and store them in a Dataset with PyTorch format.

```
import numpy as np
from datasets import Dataset


seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")
```

To print summary statistics for the GPU utilization and the training run with the Trainer we define two helper functions:

```
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
```

When a model is loaded to the GPU the kernels are also loaded, which can take up 1-2GB of memory.
To see how much it is we load a tiny tensor into the GPU which triggers the kernels to be loaded as well.

```
import torch


torch.ones((1, 1)).to("cuda")
print_gpu_utilization()
```

### Loading Model

First, we load the bert-large-uncased model.
We load the model weights directly to the GPU so that we can check how much space just the weights use.

```
from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print_gpu_utilization()
```

Note that on newer GPUs a model can sometimes take up more space since the weights are loaded in an optimized fashion.
That speeds up the usage of the model.

### Memory utitlisation at vanilla Training

Now we can start training the model and see how the GPU memory consumption changes.
First, we set up a few standard training arguments:

```
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```
Let’s use the Trainer and train the model without using any GPU performance optimization techniques and a batch size of 4:

```
from transformers import TrainingArguments, Trainer, logging

logging.set_verbosity_error()

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```
We see that already a relatively small batch size almost fills up our GPU’s entire memory. **However, a larger batch size can often result in faster model convergence or better end performance.** 
** So ideally we want to tune the batch size to our model’s needs and not to the GPU limitations. ** What’s interesting is that we use much more memory than the size of the model.
To understand a bit better why this is the case let’s have a look at a model’s operations and memory needs.

### Understanding given Bert Model

Transformers architecture includes 3 main groups of operations grouped below by compute-intensity.

- Tensor Contractions

Linear layers and components of Multi-Head Attention all do batched matrix-matrix multiplications. These operations are the most compute-intensive part of training a transformer.

- Statistical Normalizations

Softmax and layer normalization are less compute-intensive than tensor contractions, and involve one or more reduction operations, the result of which is then applied via a map.

- Element-wise Operators

These are the remaining operators: biases, dropout, activations, and residual connections. These are the least compute-intensive operations.

This summary is derived from [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)

### Anatomy of Weigts/Parameters of a Model

 There are many components of weights that use GPU memory during training. The components on GPU memory are the following:

- model weights
- optimizer states
- gradients
- forward activations saved for gradient computation
- temporary buffers
- functionality-specific memory

A typical model trained in mixed precision with AdamW requires 18 bytes per model parameter plus activation memory.
For inference there are no optimizer states and gradients, so we can subtract those. And thus we end up with 6 bytes per model parameter for mixed precision inference, plus activation memory.

Let’s look at the details.

#### Model Weights:

- 4 bytes * number of parameters for fp32 training
- 6 bytes * number of parameters for mixed precision training (maintains a model in fp32 and one in fp16 in memory)

#### Optimizer States:

- 8 bytes * number of parameters for normal AdamW (maintains 2 states)
- 2 bytes * number of parameters for 8-bit AdamW optimizers like [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- 4 bytes * number of parameters for optimizers like SGD with momentum (maintains only 1 state)

#### Gradients

- 4 bytes * number of parameters for either fp32 or mixed precision training (gradients are always kept in fp32)

#### Forward Activations

- size depends on many factors, the key ones being sequence length, hidden size and batch size.
- There are the input and output that are being passed and returned by the forward and the backward functions and the forward activations saved for gradient computation.

#### Temporary Memory

- Additionally, there are all kinds of temporary variables which get released once the calculation is done, but in the moment these could require additional memory and could push to OOM.
- Therefore, when coding it’s crucial to think strategically about such temporary variables and sometimes to explicitly free those as soon as they are no longer needed.

#### Functionality-specific memory

- Then, your software could have special memory needs. For example, when generating text using beam search, the software needs to maintain multiple copies of inputs and outputs.

#### forward vs backward Execution Speed

- For convolutions and linear layers there are 2x flops in the backward compared to the forward, which generally translates into ~2x slower (sometimes more, because sizes in the backward tend to be more awkward).
- **Activations are usually bandwidth-limited**, and it’s typical for an activation to have to read more data in the backward than in the forward (e.g. activation forward reads once, writes once, activation backward reads twice, gradOutput and output of the forward, and writes once, gradInput).
