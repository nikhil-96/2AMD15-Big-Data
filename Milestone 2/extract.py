dataLog = []
with open('submission-18ede232eb354ca4aa90b401a3d461ea.spark-submit.log', 'rt') as f:
    data = f.readlines()
for line in data:
    if '>>>>' in line:
        print(line)
        #dataLog.append(line)
print(dataLog)