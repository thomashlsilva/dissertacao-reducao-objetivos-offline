import numpy as np
from reduction_lap_score.main import main
from multiprocessing import Pool
import datetime
import winsound

# get the start datetime
st = datetime.datetime.now()

"""
# DTLZ1
vec_5obj = np.array([0, 1, 2, 3, 4])

vec_10obj = np.array([1, 0, 2, 4, 3, 5, 6, 7, 8, 9])

vec_15obj = np.array([0, 3, 1, 4, 7, 5, 2, 6, 9, 10, 8, 11, 12, 13, 14])

vec_20obj = np.array([3, 5, 6, 0, 8, 1, 4, 7, 2, 12, 11, 9, 10, 13, 15, 14, 17, 16, 18, 19])

vec_30obj = np.array([1, 11, 17, 12, 9, 14, 2, 6, 5, 16, 15, 4, 19, 20, 18,
                      13, 7, 23, 10, 21, 3, 0, 25, 22, 24, 8, 26, 27, 28, 29])

# DTLZ2
vec_5obj = np.array([1, 0, 2, 3, 4])

vec_10obj = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9])

vec_15obj = np.array([2, 0, 1, 3, 6, 7, 4, 5, 8, 9, 10, 11, 12, 14, 13])

vec_20obj = np.array([1, 4, 3, 2, 0, 5, 8, 7, 6, 9, 10, 11, 13, 12, 14, 15, 16, 17, 19, 18])

vec_30obj = np.array([3, 4, 18, 14, 10, 1, 5, 7, 2, 13, 17, 11, 19, 6, 9, 15, 12, 0, 8,
                      16, 21, 20, 22, 23, 24, 25, 26, 27, 28, 29])

# DTLZ3
vec_5obj = np.array([1, 0, 2, 3, 4])

vec_10obj = np.array([3, 0, 1, 2, 4, 6, 5, 7, 8, 9])

vec_15obj = np.array([2, 1, 0, 5, 3, 4, 7, 6, 8, 10, 9, 11, 12, 13, 14])

vec_20obj = np.array([5, 0, 2, 4, 1, 3, 7, 6, 8, 10, 9, 11, 13, 14, 12, 15, 16, 17, 19, 18])

vec_30obj = np.array([4, 9, 3, 2, 1, 5, 10, 7, 6, 14, 13, 20, 8, 17, 22, 15, 11, 16,
                      0, 23, 12, 19, 18, 21, 24, 25, 26, 27, 28, 29])

# DTLZ4
vec_5obj = np.array([2, 0, 3, 4, 1])

vec_10obj = np.array([2, 1, 0, 6, 4, 9, 3, 8, 5, 7])

vec_15obj = np.array([12, 14, 1, 9, 10, 8, 0, 3, 13, 5, 4, 2, 7, 11, 6])

vec_20obj = np.array([10, 13, 4, 15, 17, 1, 9, 12, 16, 0, 19, 8, 11, 18, 7, 14, 5, 3, 6, 2])

vec_30obj = np.array([28, 25, 20, 14, 3, 16, 17, 12, 13, 7, 23, 10, 22, 27, 29, 9, 18,
                      11, 0, 4, 5, 24, 21, 15, 8, 2, 19, 1, 26, 6])

# DTLZ5
vec_5obj = np.array([1, 2, 0, 3, 4])

vec_10obj = np.array([5, 4, 3, 2, 0, 1, 6, 7, 8, 9])

vec_15obj = np.array([8, 5, 0, 3, 1, 7, 6, 4, 2, 9, 11, 10, 12, 13, 14])

vec_20obj = np.array([3, 1, 6, 12, 2, 7, 5, 8, 9, 10, 4, 0, 16, 11, 13, 17, 14, 15, 18, 19])

vec_30obj = np.array([0, 16, 20, 1, 21, 19, 22, 3, 11, 9, 15, 2, 7, 18,
                      6, 8, 14, 5, 13, 23, 17, 10, 12, 4, 25, 26, 24, 27, 28, 29])

# DTLZ6
vec_5obj = np.array([0, 1, 2, 3, 4])

vec_10obj = np.array([2, 1, 0, 3, 4, 5, 6, 7, 8, 9])

vec_15obj = np.array([1, 3, 5, 0, 2, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14])

vec_20obj = np.array([2, 3, 1, 0, 6, 4, 9, 8, 5, 11, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19])

vec_30obj = np.array([0, 15, 10, 12, 7, 4, 11, 14, 5, 2, 9, 6, 13, 1, 8,
                      3, 16, 19, 18, 17, 21, 20, 22, 24, 23, 25, 26, 27, 28, 29])
"""
# DTLZ7
vec_5obj = np.array([4, 1, 3, 2, 0])

vec_10obj = np.array([9, 7, 8, 3, 2, 6, 1, 4, 0, 5])

vec_15obj = np.array([14, 2, 8, 12, 11, 0, 13, 7, 3, 4, 5, 9, 10, 6, 1])

vec_20obj = np.array([19, 2, 15, 12, 5, 1, 14, 0, 17, 3, 8, 4, 11, 6, 13, 9, 7, 10, 16, 18])

vec_30obj = np.array([29, 15, 3, 16, 24, 0, 23, 12, 18, 5, 4, 26, 17, 27, 20, 
                      21, 28, 7, 10, 19, 22, 1, 14, 11, 13, 8, 2, 9, 6, 25])


# 5 objective
obj5_to4 = 5, 4, vec_5obj.copy()
obj5_to3 = 5, 3, vec_5obj.copy()
obj5_to2 = 5, 2, vec_5obj.copy()

# 10 objective
obj10_to8 = 10, 8, vec_10obj.copy()
obj10_to6 = 10, 6, vec_10obj.copy()
obj10_to4 = 10, 4, vec_10obj.copy()
obj10_to2 = 10, 2, vec_10obj.copy()

# 15 objective
obj15_to12 = 15, 12, vec_15obj.copy()
obj15_to9 = 15, 9, vec_15obj.copy()
obj15_to6 = 15, 6, vec_15obj.copy()
obj15_to3 = 15, 3, vec_15obj.copy()
obj15_to2 = 15, 2, vec_15obj.copy()

# 20 objective
obj20_to16 = 20, 16, vec_20obj.copy()
obj20_to12 = 20, 12, vec_20obj.copy()
obj20_to8 = 20, 8, vec_20obj.copy()
obj20_to4 = 20, 4, vec_20obj.copy()
obj20_to2 = 20, 2, vec_20obj.copy()

# 30 objective
obj30_to25 = 30, 25, vec_30obj.copy()
obj30_to20 = 30, 20, vec_30obj.copy()
obj30_to15 = 30, 15, vec_30obj.copy()
obj30_to10 = 30, 10, vec_30obj.copy()
obj30_to5 = 30, 5, vec_30obj.copy()
obj30_to2 = 30, 2, vec_30obj.copy()

if __name__ == '__main__':
    n_obj = int(input("Enter the number of original objectives: "))
    with Pool(4) as pool:  # four parallel jobs
        if n_obj == 5:
            pool.map(main, (obj5_to4, obj5_to3, obj5_to2))
        elif n_obj == 10:
            pool.map(main, (obj10_to8, obj10_to6, obj10_to4, obj10_to2))
        elif n_obj == 15:
            pool.map(main, (obj15_to12, obj15_to9, obj15_to6, obj15_to3, obj15_to2))
        elif n_obj == 20:
            pool.map(main, (obj20_to16, obj20_to12, obj20_to8, obj20_to4, obj20_to2))
        elif n_obj == 30:
            pool.map(main, (obj30_to25, obj30_to20, obj30_to15, obj30_to10, obj30_to5, obj30_to2))
        else:
            print("Invalid number of objectives!")

        # get the end datetime
        et = datetime.datetime.now()
        # get execution time
        elapsed_time = et - st
        print('Start datetime:', st, 'seconds')
        print('End datetime:', et, 'seconds')
        print('Execution time:', elapsed_time, 'seconds')

        duration = 750  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
