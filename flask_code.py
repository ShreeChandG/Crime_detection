import numpy as np
import os
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


app = Flask(__name__)

mobilenet_v2 = MobileNetV2(
    input_shape=(64, 64, 3),
    alpha=1.0,
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=14,
    classifier_activation='softmax'
)

for layer in mobilenet_v2.layers:
    layer.trainable = False

model = Sequential()
model.add(mobilenet_v2)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(14, activation='softmax'))  # Use the same number of classes as defined in MobileNetV2

# Load the pre-trained weights
model.load_weights("crime.h5")

# Compile the model after loading weights
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

index = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
         'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting',
         'Shoplifting', 'Stealing', 'Vandalism']


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x, axis=0)
        print(x)
        y = model.predict(x)
        preds = np.argmax(y, axis=1)
        print("prediction", preds)
        predicted_label = index[preds[0]]
        
        #img_path = f'/uploads/{f.filename}'
        text = "The classified crime is: " + str(predicted_label)
        return text
    
    
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug = False, threaded = False)
