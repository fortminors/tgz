from typing import List

def multiplicate(a: List[int]):
    num_items = len(a)

    if (num_items <= 1):
        return [0] * num_items

    prod = 1
    num_zeros = 0
    zero_pos = None
    out = []

    # Accumulating items product
    for i in range(num_items):
        if (a[i] == 0):
            num_zeros += 1
            zero_pos = i
        else:
            prod *= a[i]

        out.append(0)

    if (num_zeros == 1):
        out[zero_pos] = prod
        return out
    elif (num_zeros > 1):
        return out

    for i in range(num_items):
        out[i] = prod / a[i]

    return out
