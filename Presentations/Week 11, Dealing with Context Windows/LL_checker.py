import pandas as pd
import numpy as np

your_df = pd.read_pickle('path/to/your/cooc/df')
cells_to_check = pd.DataFrame(np.random.randint(0, len(your_df), size = (50, 2)))

for i, (row, col) in cells_to_check.iterrows():
    real_LL = #I will insert your LL calculation script here
    print('Real LL: %s, Your LL: %s, They are equal: %s' %
          (real_LL, df.ix[row, col], real_LL == df.ix[row, col]))
