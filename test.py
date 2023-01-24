a = [i for i in range(10)]

a[1:5] = [2*i for i in a[1:5]]
print(a)