import sys
import os
import csv

if len(sys.argv) > 1:
  path_folder = sys.argv[1]
else:
  path_folder = "/content/drive/My Drive"
# path_csvFiles = [os.path.join(path_folder, file) for file in ["train.csv", "val.csv", "test.csv"] ]
csvFiles = ["train.csv", "val.csv", "test.csv"]

fout = open( os.path.join(path_folder, "list.txt"), "w" )
for fileName in csvFiles:
  path_file = os.path.join(path_folder, fileName)
  if not os.path.isfile(path_file):
    print("%s does not exist. \n" % (path_file))
    continue
  fout.write(fileName.split('.')[0]+'\n')
  with open(path_file) as fin:
    csv_reader = csv.reader(fin, delimiter=',')
    lineCount = 0
    itemDict = {}
    for line in csv_reader:
      if lineCount == 0: # first line
        lineCount += 1
        continue
      city = line[0].split('/')[-2]
      if city not in itemDict:
        itemDict[city] = 1
      else:
        itemDict[city] += 1
      lineCount += 1

  for city in itemDict:
    fout.write("\t%s %d\n" % (city, itemDict[city]))

fout.close()
print("File saved to %s" % (os.path.join(path_folder, "list.txt")))