
import matplotlib.pyplot as plt 
import csv
import sys

input_file = sys.argv[1]
x = [] 
y = [] 
  
with open(input_file,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines: 
        x.append(row[1]) 
        y.append(row[2]) 
        print(row[1],row[2])
plt.scatter(x, y, alpha=0.7)
plt.title('BiolekJogelkarMC') 
plt.show() 
