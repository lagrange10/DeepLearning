from os import chdir
import pickle
chdir("src\posenet\pretrained_models")
file = open('./places-googlenet.pickle', "rb")
weights = pickle.load(file, encoding="bytes")
print(weights.keys())