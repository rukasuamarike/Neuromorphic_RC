import csv
import sys

input_file = sys.argv[1]
data = None
with open(input_file, "r") as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

parsed_data = data[1:]
cleaned_data=[]

for row in parsed_data:
    # Split on whitespace, removing extra spaces
    values = row[0].split()
    cleaned_data.append(values)
print(cleaned_data)
print(data[0][0].split())
with open(input_file[:-4] + "_output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data[0][0].split())  # Write header row
    writer.writerows(cleaned_data)
