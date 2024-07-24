def common_count(list1, list2):
    count = 0
    common = []
    for i in list1:
        if i in list2:
            count += 1
            common.append(i)
            continue
    return count, common

if __name__ == '__main__':
    list1 = list(input("Enter list1 ").split())
    list2 = list(input("Enter list2 ").split())
    count, common = common_count(list1, list2)
    print("Count of common elements: ", count)
    print("Common elements: ", common)