import csv
import os

folder_name = r"C:\Users\USER\PycharmProjects\documents22"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open(r'C:\Users\USER\Downloads\IR\wikIR1k\wikIR1k\collection.tsv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        print(row[0])
        file_id = row[0]
        print(file_id)
        content = row[1]
        print(content)

        file_path = os.path.join(folder_name, f"{file_id}.txt")
        with open(file_path, "w") as outfile:
            outfile.write(content)

