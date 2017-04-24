# def quicksort(arr):
# 	if len(arr)<=1:
# 		return arr
# 	pivot = arr[len(arr)/2]
# 	left = [x for x in arr if x < pivot]
# 	print("------------------------")
# 	print("left elements",left)
# 	middle = [x for x in arr if x == pivot]
# 	print("------------------------")
# 	print("middle elements",middle)
# 	right = [x for x in arr if x > pivot]
# 	print("------------------------")
# 	print("right elements",right)
# 	return quicksort(left) + middle + quicksort(right) 

# print quicksort([9,8,7,6,5,4,3,2,1])

import numpy as np
test1 = np.array([1,2])
print(np.shape(test1))
test2 = np.array([[1001,1002],[3,4]])
print(np.shape(test2))
# c = np.max(test2, axis=1).reshape(-1,1)
# print (np.shape(c))
test3 = np.array([[-1001,-1002]])
print(np.shape(test3))