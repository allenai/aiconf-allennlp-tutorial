import csv
import pathlib
import os
import re

import tqdm

DATA_ROOT = pathlib.Path("data")
BBC_ROOT = DATA_ROOT / 'bbc'

train = []
validate = []
test = []

for category in os.listdir(BBC_ROOT):
    path = BBC_ROOT / category
    if os.path.isdir(path):
        for fn in os.listdir(path):
            with open(path / fn, errors="ignore") as f:
                text = f.read()

            lines = text.split("\n")
            lines = [line for line in lines if line]
            lines = lines[:2]
            text = "\n\n".join(lines)

            number = int(fn[:3])

            if number % 5 < 3:
                train.append((category, text))
            elif number % 5 == 3:
                validate.append((category, text))
            else:
                test.append((category, text))

for fn, instances in [(DATA_ROOT / 'bbc-train.csv', train),
                      (DATA_ROOT / 'bbc-validate.csv', validate),
                      (DATA_ROOT / 'bbc-test.csv', test)]:
    print(fn)

    with open(fn, 'wt') as f:
        writer = csv.writer(f)
        for row in tqdm.tqdm(instances):
            writer.writerow(row)
