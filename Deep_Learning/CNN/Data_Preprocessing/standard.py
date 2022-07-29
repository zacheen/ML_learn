import numpy as np

# cmin cmax 測試
array = np.array(range(2,14)).astype("float32")
print(array)
Cmin = 3
Cmax = 12*1.15 # 小數值*1.15 看起來還好 但是大數值就會差很多
print((array-Cmin) / (Cmax-Cmin))

# Cmax 乘上某個數值 不是很穩定
# 小數值*1.15 看起來還好 但是大數值就會差很多
array = np.array(range(9002,9014)).astype("float32")
print(array)
Cmin = 9003
Cmax = 9012*1.15 
array = (array-Cmin) / (Cmax-Cmin)
print(array)

####################################################
