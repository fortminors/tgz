from pathlib import Path
from multiprocessing import Pool, Value

import numpy as np

mean = None
std = None
counter = None

def InitWorker(m, s, c):
    global mean
    global std
    global counter

    mean = m
    std = s
    counter = c

def SpectogramToImage(spec):
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec = spec - spec.min()
    spec = spec / (spec.max() + 1e-6)
    return spec

def ComputeImageStats(image_path):
    global mean
    global std
    global counter
    
    spectogram = np.load(image_path)

    image = SpectogramToImage(spectogram)

    with mean.get_lock():
        mean.value += image.mean()

    with std.get_lock():
        std.value += image.std()
    
    with counter.get_lock():
        counter.value +=1

        if (counter.value % 1000 == 0):
            print(f"{counter.value} done")

if __name__ == '__main__':
    mean = Value('d', 0.0)
    std = Value('d', 0.0)
    counter = Value('i', 0)

    image_paths = list(Path('train1/train').rglob('*.npy')) + list(Path('train2/train').rglob('*.npy')) + list(Path('val/val').rglob('*.npy'))

    with Pool(initializer=InitWorker, initargs=(mean, std, counter, )) as pool:
        pool.map(ComputeImageStats, image_paths)

    mean = mean.value / counter.value
    std = std.value / counter.value

    print(f"{mean = } , {std = }")
