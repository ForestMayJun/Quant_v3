import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame([np.random.rand(10), np.random.rand(10)]).T
# plt.plot(df.iloc[:, 0])
plt.plot(np.random.rand(10))
plt.show()