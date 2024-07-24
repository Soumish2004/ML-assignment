def count(string):
    vowels = 'aeiou'
    vc = 0
    cc = 0
    for i in string:
        if i.lower() in vowels:
            vc += 1
        else:
            cc += 1
    return vc, cc

if __name__ == '__main__':
    input_string = input("Enter a String ")
    vc, cc = count(input_string)
    print("Vowels: ", vc)
    print("Consonants: ", cc)