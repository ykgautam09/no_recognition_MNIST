from flask import request, render_template, Flask
from PIL import Image
import pickle
import png
import numpy as np

app = Flask(__name__)
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['POST', 'GET'])
def home():
    result_c = ''
    if request.method == 'POST':
        image_uploaded = request.files['image_file']

        # filepath='testImages/mnist_data4.png'
        img = np.array(Image.open(image_uploaded))

        result_c = model.predict([img.reshape(784)])[0]
        print(result_c)
    return render_template('home.html', msg=result_c)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
