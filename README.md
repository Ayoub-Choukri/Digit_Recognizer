# Preamble
**Author**: Ayoub CHOUKRI
**Date**: 

# How to use this repository

In order, to launch the website, you need to follow the following steps:

1. Clone the repository through the following command:
```bash
git clone 
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the website:
```bash
source ./Site/lanch.sh
```
Toutefois, il faut remplacer le nom de l'environement "Personnel' par le nom de votre environement.

# Description of the repository

In this repository, i tried to build a Flask Website that contains a digit recognizer. The website is composed of two pages:

1. The home page: This page allow to select which model to use in order to predict the digits.

2. The prediction page: This page allow to draw a digit and predict it using the selected model.

# Structure of the repository

The repository is structured as follows:

- The `Site` folder contains the website and the APIs that allow to predict the digits.

- The `Main` folder contains a file names train_main.py that allow to train the models.

- The `Trained_Models` folder contains the trained models.

- The `Modules` folder contains the modules that are used in the website and in the training of the models.

