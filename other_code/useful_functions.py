import numpy as np

def get_mean_std_emnist(fed_dataset):
    mean = np.zeros((28, 28), dtype=np.float64)
    meansq = np.zeros((28, 28), dtype=np.float64)
    count = 0
    for client_id in fed_dataset.client_ids:
        example_dataset = train_data.create_tf_dataset_for_client(client_id)
        x = next(example_dataset.batch(1000).as_numpy_iterator())['pixels'].astype(np.float64)
        mean = mean * count / (count + x.shape[0]) + x.mean(axis=0) * x.shape[0] / (count + x.shape[0])
        meansq = meansq * count / (count + x.shape[0]) + (x**2).mean(axis=0) * x.shape[0] / (count + x.shape[0])
        count += x.shape[0]
    return mean, np.sqrt(meansq - mean**2)