import numpy as np
aa = np.arange(15)
a = aa.reshape(5,3)
print(a[:: -1])

y = np.append(aa, (100))
print(y)