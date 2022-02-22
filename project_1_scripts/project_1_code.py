import numpy as np
import pandas as pd

url = "https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data"
df = pd.read_csv(url)
print(df)

raw_data = df.values
print(raw_data)

cols = range (1,8) # exclude column 0 because
  # row numbers are not needed as attributes
X = raw_data[:, cols]
print(X)

attribute_names = np.asarray(df.columns[cols])
print(attribute_names)
