# Ce code va créer l'API qui va encapsuler le modèle de machine learning.

import numpy as np
from flask import Flask, render_template, request, jsonify
import torch
import os
import sys
import base64
import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

Module_Path1 = "../Modules/Models"
Module_Path2 = "../Modules/"

if Module_Path1 not in sys.path:
    sys.path.append(Module_Path1)

if Module_Path2 not in sys.path:
    sys.path.append(Module_Path2)


from resnet import *
from preprocessing import *
from train import *
import requests



WEBSITE_API_URL = "http://127.0.0.1:5000"

app = Flask(__name__)


Model_Name = None
Model_Paths = {'resnet' : "../Trained_Models/Mnist_Resnet18.pth"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_models = {}
model_to_use = None

def Load_Resnet(model_name):
    global Model_Paths
    global device
    global loaded_models
    global model_to_use

    if model_name not in loaded_models:
        loaded_models[model_name] = Load_Model( Model_Paths[model_name] ,device)
        loaded_models[model_name].to(device)
        model_to_use = loaded_models[model_name]
        print(f"Model {model_name} loaded successfully")



@app.route('/Specify_Model', methods=['Post'])
def Specify_Model():
    global loaded_models
    global device
    global Model_Name


    data = request.get_json()

    if Model_Name is None:
        Model_Name = data['model_name']
        Load_Model_API(Model_Name)
        return jsonify({'message': f'Model {Model_Name} loaded successfully'}), 200

@app.route('/Change_Model', methods=['Post'])
def Change_Model():
    global loaded_models
    global device

    data = request.get_json()

    Model_Name = data['model_name']
    Load_Model_API(Model_Name)
    
    return jsonify({'message': f'Model {Model_Name} loaded successfully'}), 200


@app.route('/Load<model_name>', methods=['GET'])
def Load_Model_API(model_name):
    global loaded_models
    global Model_Paths
    global deviceLoad_Model

    if model_name == "resnet" : 
        Load_Resnet(model_name)
        return jsonify({'message': f'Model {model_name} loaded successfully'}), 200
    else:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    




@app.route('/Predict', methods=['POST'])
def Predict():

    global loaded_models
    global device
    global model_to_use


    if model_to_use is None:
        print(f"Model_Name : {Model_Name}")
        assert False, "No model loaded"

    else:

        data = request.get_json()

        image_data = data['image']

        # Décoder l'image en base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Prétraitement de l'image

        image = Preprocess_Image(image)

        image = image.unsqueeze(0).to(device)

        model_to_use.eval()

        with torch.no_grad():

            image = image.float()
            image = image.to(device)
            output = model_to_use(image)
            # Apply softmax
            output = torch.nn.functional.softmax(output, dim=1)
            # Round to 3 decimal places
            output =  torch.round(output * 1000) / 1000

            _, predicted = torch.max(output, 1)
            output = output[0]
            return jsonify({'prediction': predicted.item(),'probs': output.tolist()}), 200
        
if __name__ == '__main__':

    app.run(debug=True, port= 5001)







    
    





    

