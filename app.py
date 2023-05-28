import pickle
from flask import Flask,jsonify,request,app,render_template,Response
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            d1 = request.json['data']
            print('data is:     ', d1)

            model = pickle.load(open('pipe.pkl','rb'))

            querry = np.array(list(d1.values())).reshape(1,-1)
            result = model.predict(querry)
            return jsonify(result)
        
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)


if __name__ == "__main__":
    app.run(debug= True)