import json


cls_list = ['fat', 'littleFat', 'normal', 'thin', 'veryFat']
arrJson = []
cal = [0.49532357,0.49532357,0.49532357,0.49532357]
for i in cal :
    i_int = "{0:.3f}".format(i)
    testJson = {'per':i_int}
    arrJson.append(testJson)

Res = json.dumps(arrJson)
print(Res)




