# Dataset Details

For each dataset, load the following files:
- `{dataset}_client_ids_train.csv` and `{dataset}_client_ids_test.csv`: save with a header
```
"train_ids"
"id1"
"id2"
...
```
To load, run:
```
    lst = pd.read_csv(client_list_fn, dtype=str).to_numpy().reshape(-1).tolist()
```
To save, run:
```
    selected_client_ids = # list of strings
    args = dict(index=False, header=True, quoting=csv.QUOTE_ALL)
    pd.Series(selected_client_ids, name='train_ids').to_csv(f'dataset_statistics/{dataset}_client_ids_train.csv', **args)
```
- `{dataset}_client_sizes_train.csv` and `{dataset}_client_sizes_test.csv`: client id and size in a csv
```
client_id,size
id1,123
id2,234
...
```
To load, run:
```python
    client_sizes = pd.read_csv(sizes_filename, index_col=0, squeeze=True, dtype='string').to_dict()
```
To save, run:
```
    TODO
```
- [optional] `{dataset}_mean.csv` and `{dataset}_std.csv`: 
To load, run:
```
    torch.from_numpy(pd.read_csv(mean_filename).to_numpy().astype(np.float32))
```
To save, run:
```

```
For celeba:
```
    mean = [128.48366490779233, 107.9964640295695, 97.20552669177295]
    std = [78.43797746627817, 73.26972637336158, 73.0293954663978]
```


# Details:

## Stack Overflow
- at least 100 training sequences and 10 testing sequences: there are 116857 of such clients
- sample 1000 of them
```
selected_ids_to_train = list(selected_ids_to_train_size.keys())
import random
rng = random.Random(1) 
selected_ids_to_test = rng.sample(selected_ids_to_train, 1000)  # sample 1000 clients for testing
```

## EMINST
- >= 100 training points and >= 25 testing points: there are 1114 of such clients

## GLDv2
- consider all train clients from TFF with at least 50 data points: there are 823 of them
- split their data 50:50 into train and test

## CelebA