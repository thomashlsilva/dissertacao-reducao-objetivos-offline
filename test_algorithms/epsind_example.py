import numpy as np

'''
def eps_ind_add(A, B):
    aux = []
    for i in range(len(B)):
        aux.append(np.sum(np.abs(A - np.tile(B[i], (A.shape[0], 1))), axis=1))
    return max(np.min(aux, axis=0))
'''


def eps_ind_mult(A, B):
    aux = []
    for i in range(len(B)):
        aux.append(np.min(np.max(A / np.tile(B[i], (A.shape[0], 1)), axis=1)))
    return max(aux)


def eps_ind_add(A, B):
    aux = []
    for i in range(len(B)):
        aux.append(np.min(np.max(A - np.tile(B[i], (A.shape[0], 1)), axis=1)))
    return max(aux)


def main():
    A = [
        np.array([[4, 7], [5, 6], [7, 5], [8, 4], [9, 2]]),
        np.array([[4, 7], [5, 6], [7, 5], [8, 4]]),
        np.array([[6, 8], [7, 7], [8, 6], [9, 5], [10, 4]]),
        np.array([[1, 3], [2, 2], [3, 1]])
    ]

    for i in range(4):
        for j in range(4):
            print(f'A{i + 1} x A{j + 1}')
            # result = eps_ind_mult(A[i], A[j])
            result = eps_ind_add(A[i], A[j])
            print(result)


if __name__ == '__main__':
    main()
