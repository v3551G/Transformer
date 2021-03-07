# -*- coding: utf-8 -*-
a = [1,2,3]
b = [4,5,6,7]
c = zip(b,a)
for x, y in c:
    print(x, '_', y)
