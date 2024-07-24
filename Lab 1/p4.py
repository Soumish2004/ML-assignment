import numpy as np

def transpose(matrix1):
    result = []
    for i in range(matrix1.shape[1]):
        row = []
        for j in range(matrix1.shape[0]):
            row.append(matrix1[j, i])
        result.append(row)
    return np.array(result).reshape(matrix1.shape[1], matrix1.shape[0])

if __name__ == '__main__':
    matrix = list(map(int,input().split()))
    r1 = int(input("Enter number of rows for matrix: "))
    c1 = int(input("Enter number of columns for matrix: "))
    matrix = np.array(matrix).reshape(r1, c1)
    print("Original Matrix:\n", matrix)

    print("Transposed Matrix:\n",transpose(matrix))