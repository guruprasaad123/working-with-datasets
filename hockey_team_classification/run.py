import os
import csv
from pathlib import Path
import random
import time

if os.path.exists('./dataset') == False:
    raise Exception('Download Dataset')

seed_numbers = range(1, 10)

ids = [f for f in os.listdir('./dataset')]

with open("submissions/submission.{}.csv".format(int(time.time())), 'w', newline='') as csvfile:
    fieldnames = ['id', 'team_a', 'team_b', 'time (ms)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for id in ids:
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        random.shuffle(numbers)
        aNumber = random.choice(seed_numbers)
        bNumber = 10 - aNumber
        team_a = []
        team_b = []

        images = [f for f in os.listdir('./dataset/{}'.format(id)) if f.endswith('.jpg')]

        if len(images) != 10:
            raise Exception("Invalid Dataset {}".format(id))

        ms = int(round(time.time() * 1000))

        # Execute your code to sort players here

        ms = int(round(time.time() * 1000)) - ms

        writer.writerow({'id': id, 'team_a': ",".join(
                        team_a), 'team_b': ",".join(team_b), 'time (ms)': str(ms)})