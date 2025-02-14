#%%
import numpy as np

# Example arrays
a = np.array([1, 0, 1, 0])
b = np.array([0, 0, 1, 1])

# Logical OR
result = np.logical_or(a, b).astype(int)
print(result)
