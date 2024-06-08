from datasets import load_dataset


def load_data():
    dataset = load_dataset("allenai/peS2o")
    return dataset


dataset = load_data()
print(len(dataset))