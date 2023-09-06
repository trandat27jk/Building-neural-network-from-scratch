import  numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

A = np.random.randn(4, 3)
B=np.sum(A, axis=1, keepdims=True)
print(B)
