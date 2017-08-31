i=0
with open("BdD_Test", "r") as f:
    for line in f.readlines():
        i+=1
        if i%3==1:
            red,row = line.strip().split(":")
            stat0,foo = row.split(";")
            print(stat0)

