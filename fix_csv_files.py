import os
import re
from utils import fix_csv_file

files_location = './experiments_results/results/'

csv_files = [f for f in os.listdir(files_location) if re.match(r'^[^\.].*\.csv$', f)]

for csv_file in csv_files:
    fix_csv_file(os.path.join(files_location, csv_file), 'architecture')

