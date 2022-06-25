from tqdm import tqdm
from collections import defaultdict
import numpy as np

from data.sequoia import WeedMapDataset

def count_classes(root, folders):
    channels = ['NDVI']
    index = WeedMapDataset.build_index(
        root,
        macro_folders=folders,
        channels=channels,
    )
    sq = WeedMapDataset(root,
                        transform=lambda x: x,
                        target_transform=lambda x: np.array(x),
                        index=index,
                        channels=channels,
                        return_path=True
                        )
    # loop through images
    class_counts = defaultdict(lambda: 0)
    for input, target, additional in tqdm(sq):
        d, counts = np.unique(target, return_counts=True)
        for v, c in zip(d, counts):
            class_counts[v] += c
    print(class_counts)
    values = np.array(list(class_counts.values()))
    print(values / np.sum(values))


if __name__ == '__main__':
    f = ['005']
    r = "./dataset/processed/Sequoia"
    count_classes(r, f)