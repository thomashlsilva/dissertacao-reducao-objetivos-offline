from new_main import main
from multiprocessing import Pool
import pandas as pd
import datetime
import winsound


number_dtlz = 7
# get the start datetime
st = datetime.datetime.now()

eigenvectors_5obj = pd.read_csv(f"./eigenvectors/dtlz{number_dtlz}/5obj_eigenvectors.csv", header=None).to_numpy()
eigenvectors_10obj = pd.read_csv(f"./eigenvectors/dtlz{number_dtlz}/10obj_eigenvectors.csv", header=None).to_numpy()
eigenvectors_15obj = pd.read_csv(f"./eigenvectors/dtlz{number_dtlz}/15obj_eigenvectors.csv", header=None).to_numpy()
eigenvectors_20obj = pd.read_csv(f"./eigenvectors/dtlz{number_dtlz}/20obj_eigenvectors.csv", header=None).to_numpy()
eigenvectors_30obj = pd.read_csv(f"./eigenvectors/dtlz{number_dtlz}/30obj_eigenvectors.csv", header=None).to_numpy()

# 5 objective
# obj5_to5 = 5, 5, eigenvectors_5obj
obj5_to4 = 5, 4, eigenvectors_5obj.copy()
obj5_to3 = 5, 3, eigenvectors_5obj.copy()
obj5_to2 = 5, 2, eigenvectors_5obj.copy()

# 10 objective
obj10_to8 = 10, 8, eigenvectors_10obj.copy()
obj10_to6 = 10, 6, eigenvectors_10obj.copy()
obj10_to4 = 10, 4, eigenvectors_10obj.copy()
obj10_to2 = 10, 2, eigenvectors_10obj.copy()

# 15 objective
obj15_to12 = 15, 12, eigenvectors_15obj.copy()
obj15_to9 = 15, 9, eigenvectors_15obj.copy()
obj15_to6 = 15, 6, eigenvectors_15obj.copy()
obj15_to3 = 15, 3, eigenvectors_15obj.copy()
obj15_to2 = 15, 2, eigenvectors_15obj.copy()

# 20 objective
obj20_to16 = 20, 16, eigenvectors_20obj.copy()
obj20_to12 = 20, 12, eigenvectors_20obj.copy()
obj20_to8 = 20, 8, eigenvectors_20obj.copy()
obj20_to4 = 20, 4, eigenvectors_20obj.copy()
obj20_to2 = 20, 2, eigenvectors_20obj.copy()

# 30 objective
obj30_to25 = 30, 25, eigenvectors_30obj.copy()
obj30_to20 = 30, 20, eigenvectors_30obj.copy()
obj30_to15 = 30, 15, eigenvectors_30obj.copy()
obj30_to10 = 30, 10, eigenvectors_30obj.copy()
obj30_to5 = 30, 5, eigenvectors_30obj.copy()
obj30_to2 = 30, 2, eigenvectors_30obj.copy()

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
