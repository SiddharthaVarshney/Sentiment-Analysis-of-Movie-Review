# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename2 = 'movie-genre-mnb-model.pkl'
classifier2 = pickle.load(open(filename, 'rb'))
cv2 = pickle.load(open('../cv-transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict1',methods=['POST'])
def predict1():
    if request.method == 'POST':
    	message = request.form['message']
		data = [message]
		vect = cv2.transform(data).toarray()
		my_prediction = classifier2.predict(vect)
		return render_template('res2.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)