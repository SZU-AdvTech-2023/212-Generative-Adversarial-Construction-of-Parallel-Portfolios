import json
#import 
optimumFile=("TSP_optimum.json")
with open(optimumFile,'r') as f:
    optimum = json.load(f)

#print(optimum)
#print(type(optimum))
fo=open("training_index_small_1","w")
for key, value in optimum.iteritems():
    fo.write(key+"\n")
fo.close()

