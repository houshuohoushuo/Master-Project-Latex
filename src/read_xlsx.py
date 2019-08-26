from collections import Counter

import pandas as pd

file_errors_location = 'BC_Only.xlsx'
df = pd.read_excel(file_errors_location, index_col=0).to_dict()
tumor_code = df['"Tumor Code"']

labels = list(tumor_code.values())
print(Counter(labels))
