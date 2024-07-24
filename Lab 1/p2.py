import numpy as np

def check_calculate(matrix1, matrix2):
    if matrix1.shape[1] != matrix2.shape[0]:
        return False, None
    else:
        result = []
        for i in range(matrix1.shape[0]):
            row = []
            for j in range(matrix2.shape[1]):
                element = 0
                for k in range(matrix2.shape[0]):
                    element += matrix1[i, k] * matrix2[k, j]
                row.append(element)
            result.append(row)
        matrix = np.array(result)
        return True, matrix

if __name__ == '__main__':
    matrix1 = list(map(int,input().split()))
    r1 = int(input("Enter number of rows for matrix1: "))
    c1 = int(input("Enter number of columns for matrix1: "))
    matrix1 = np.array(matrix1).reshape(r1, c1)
    
    matrix2 = list(map(int,input().split()))
    r2 = int(input("Enter number of rows for matrix2: "))
    c2 = int(input("Enter number of columns for matrix2: "))
    matrix2 = np.array(matrix2).reshape(r2, c2)

    print("First Matrix:\n",matrix1)
    print("Second Matrix:\n",matrix2)

    check, result = check_calculate(matrix1, matrix2)
    if check:
        print("Result after multiplication:\n",result)
    else:
        print("Matrix multiplication not possible")
