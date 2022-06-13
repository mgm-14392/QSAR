import pandas as pd

path = '/Users/marianagonzmed/Desktop/P28845/descriptors_activity_smi_P28845.csv'
descriptors = pd.read_csv(path)
zero_cols = [ col for col, is_zero in ((descriptors == 0).sum() == descriptors.shape[0]).items() if is_zero ]
print(len(zero_cols))