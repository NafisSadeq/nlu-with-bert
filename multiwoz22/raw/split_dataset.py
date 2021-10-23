import json

with open("data.json",'r') as file:
    data=json.load(file)

test_file_list=[]
val_file_list=[]

with open("valListFile.txt",'r') as file:
    for line in file:
        val_file_list.append(line.strip())

with open("testListFile.txt",'r') as file:
    for line in file:
        test_file_list.append(line.strip())

train_set={}
val_set={}
test_set={}

for key,value in data.items():
    if(key in val_file_list):
        val_set[key]=value
    elif(key in test_file_list):
        test_set[key]=value
    else:
        train_set[key]=value

with open("train.json",'w') as file:
    json.dump(train_set,file,indent=4)

with open("val.json",'w') as file:
    json.dump(val_set,file,indent=4)

with open("test.json",'w') as file:
    json.dump(test_set,file,indent=4)
