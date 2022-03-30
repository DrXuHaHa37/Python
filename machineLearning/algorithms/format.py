def format_column(limit, arr):
    floor = limit[0]
    ceiling = limit[1]
    for a in range(len(arr)):
        if floor <= arr[a] <= ceiling:
            continue
        elif arr[a] < floor:
            arr[a] = floor
        else:
            arr[a] = ceiling
    return arr

