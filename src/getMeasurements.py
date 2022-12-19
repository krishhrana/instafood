import pandas as pd
import numpy as np

table = pd.read_html('https://www.adducation.info/how-to-improve-your-knowledge/units-of-measurement/')
table[0].to_csv('measurements.csv', index = False)