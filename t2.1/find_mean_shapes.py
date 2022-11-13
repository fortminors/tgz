
from pathlib import Path
from multiprocessing import Pool, Value

import numpy as np

widths = None
heights = None
counter = None

def InitWorker(w, h, c):
    global widths
    global heights
    global counter

    widths = w
    heights = h
    counter = c

def ComputeWidthsAndHeights(image_path):
    global widths
    global heights
    global counter
    
    spectogram = np.load(image_path)

    with heights.get_lock():
        heights.value += spectogram.shape[0]

    with widths.get_lock():
        widths.value += spectogram.shape[1]
    
    with counter.get_lock():
        counter.value +=1

        if (counter.value % 1000 == 0):
            print(f"{counter.value} done")

if __name__ == '__main__':
    widths = Value('i', 0)
    heights = Value('i', 0)
    counter = Value('i', 0)

    image_paths = list(Path('train1/train').rglob('*.npy')) + list(Path('train2/train').rglob('*.npy')) + list(Path('val/val').rglob('*.npy'))

    with Pool(initializer=InitWorker, initargs=(widths, heights, counter, )) as pool:
        pool.map(ComputeWidthsAndHeights, image_paths)

    mean_width = widths.value / counter.value
    mean_height = heights.value / counter.value

    print(f"{mean_width = } , {mean_height = }")
