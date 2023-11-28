# Dataset

*torch.utils.data.Dataset* is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

- __len__ so that len(dataset) returns the size of the dataset.

- __getitem__ to support the indexing such that dataset[i] can be used to get ith sample.

## DataLoader

*torch.utils.data.DataLoader* is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. Benefits:

- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers.

## Collate_fn

It is used to process the list of samples which later forms a batch. The batch argument is a list with all your samples. The collate_fn receives a list of tuples if your __getitem__ function from a Dataset subclass returns a tuple, or just a normal list if your Dataset subclass returns only one element.
Its main objective is to create your batch without spending much time implementing it manually. PyTorch will use default implementaiton to put batch_size examples together as you would using torch.stack (not exactly it, but it is simple like that).

If you would like to return variable-sized data, you can pad them and return a tensor in collate_fn.

```python
def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()
```

```python
DataLoader(toy_dataset, collate_fn=collate_fn, batch_size=5)
```

Why we are returning the length ?
To preserve the information of the length of each sample, to get original sequence back.

```python
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask
```

```python
def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    # Here is a generalised version for a list of jagged 2-dim tensors which have all but the first dim varying in length
    out_dims = (num, max_len, *sequences[0].shape[1:])
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask
```

Reference for [above](https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14)

```python
from functools import partial
pad_frames = partial(torch.nn.utils.rnn.pad_sequence, batch_first=True, padding_value=-1.)
dl = torch.utils.data.DataLoader(vl, batch_size=10, shuffle=True, collate_fn=pad_frames)
```

Try above one
