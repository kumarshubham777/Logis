
__author__ = 'Shubham'

import numpy as np
import itertools
lst = list(itertools.product([0, 1], repeat=5))
print(lst)
for e in lst:
    temp=[]
    cntr=1
    for i in e:
         if i==1:
             temp.append(cntr)
         cntr=cntr+1





