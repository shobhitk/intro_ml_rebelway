import sys
import hou
import numpy as np
import pandas as pd
import pickle
sys.path.insert(0, "/Users/shobhitkhinvasara/Documents/GitHub/intro_ml_rebelway/mnist/")
from nn_data_utils import DataUtils
from nn_metrics import Metrics
from nn_encoders import Encoders
from nn_activations import Activations
from nn_neural_network import NeuralNetwork

node = hou.pwd()
geo = node.geometry()

prims = geo.prims()

tmp_list = []

for prim in prims:
    tmp_list.append(prim.attribValue("mask"))
    
column_names = [f'pixel{i}' for i in range(len(tmp_list))]
    
data = pd.DataFrame([tmp_list], columns=column_names)

values = np.array(data).T
#values = values.reshape(28,28)
#values = values.ravel().T
print(values.shape)

export_path = "/Users/shobhitkhinvasara/Documents/GitHub/intro_ml_rebelway/mnist/houdini.csv"

#data.to_csv(export_path)

train = pd.read_csv("/Users/shobhitkhinvasara/Documents/GitHub/intro_ml_rebelway/mnist/train.csv")

neural_model = NeuralNetwork(train)


pickle_file = "/Users/shobhitkhinvasara/Documents/model_nn.pkl"
with open(pickle_file,"rb") as f:
    weights = pickle.load(f)
    
w1,b1,w2,b2 = [*weights][0]

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = neural_model.forward(W1, b1, W2, b2, X)
    predictions = neural_model.get_predictions(A2)
    return predictions
    
def test_prediction(index, W1, b1, W2, b2):
    current_image = values[:,index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    print("Prediction: ", prediction)
    
test_prediction(0, w1, b1, w2, b2)
