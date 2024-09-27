from flask import Flask, render_template, request, jsonify
import os
import shutil
import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import requests
import subprocess




app = Flask(__name__)

# Chemin vers le dossier de sauvegarde des images
UPLOAD_FOLDER = './Test_Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_MODEL_URL = "http://127.0.0.1:5001"
app.config['API_MODEL_URL'] = API_MODEL_URL
# Réinitialisation du dossier Test_Images
def reset_test_images_folder():
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'])
        return True
    except Exception as e:
        print(f"Erreur lors de la réinitialisation du dossier Test_Images : {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/')
# def index():
#     Ask = True
#     while Ask:
#         response = requests.get(WEBSITE_API_URL + '/wich_model')
#         if response.status_code == 200:
#             model_name = response.text
#             if model_name:
#                 Ask = False
#         else:
#             Ask = True


#     return render_template('load_model.html', model_name=model_name)

# @app.route('/wich_model', methods=['GET'])
# def wich_mode



@app.route('/drawing')
def drawing():
    # subprocess.run(["python", "api_model.py"])
    return render_template('drawing.html')

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        data = request.get_json()
        if 'image' in data:
            image_data = data['image'].split(',')[1]  # Remove header 'data:image/png;base64,'
            img_bytes = base64.b64decode(image_data)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'drawing.png')
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            return jsonify({'message': 'Image enregistrée avec succès'}), 200
        else:
            return jsonify({'error': 'Données d\'image manquantes'}), 400
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de l'image : {e}")
        return jsonify({'error': 'Erreur lors de l\'enregistrement de l\'image'}), 500
    

@app.route('/predict-image', methods=['GET'])
def predict_image():
    print(os.getcwd())  

    img = mpimg.imread('./Test_Images/drawing.png')


    img = Image.fromarray((img * 255).astype('uint8'))

    

    # Faire une requette
    
    buffered = io.BytesIO()

    img.save(buffered, format="PNG")

    image_bytes = buffered.getvalue()

    encoded_string = base64.b64encode(image_bytes).decode('utf-8')


    payload = {
        'image': encoded_string ,
        'file_name': 'drawing.png',
        'file_extension': 'png'
    }


    headers = {
        'Content-Type': 'application/json',
    }
    print('r')
    response = requests.post(f'{API_MODEL_URL}/Predictresnet', headers=headers, json = payload)

    print(response)
    return response.json()



    


@app.route('/reset-test-images', methods=['POST'])
def reset_test_images():
    if reset_test_images_folder():
        return jsonify({'message': 'Dossier Test_Images réinitialisé avec succès'}), 200
    else:
        return jsonify({'error': 'Erreur lors de la réinitialisation du dossier Test_Images'}), 500

if __name__ == '__main__':
    app.run(debug=True)
