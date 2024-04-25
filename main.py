# Importing essential libraries
import re
import numpy as np
import nltk
from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow import keras
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords_ = stopwords.words("english")
",".join(stopwords_)

model_final = load_model('model.h5')
app=Flask(__name__,template_folder='template')

def vec(review):

	review = re.sub('[^A-Za-z]', ' ', review)

	# stemming
	review = review.split(" ")
	review = [w for w in review if w != ""]
	review = [stemmer.stem(w) for w in review]

	# remove stepwords
	review = [w for w in review if w not in stopwords_]
	print(review)

	# get bag of words
	list_ = []
	for w in review:
		try:
			list_.append([w])
		except:
			pass
	test = [list_]
	print(test)
	test = np.array(test)
	test = keras.preprocessing.sequence.pad_sequences(test, 125)
	return test
@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
		if request.method == 'POST':
			review= request.form['message']
			print(review)
			pred = vec(review)
			print(pred)
			my_prediction = model_final.predict(pred)[0][0]
			print(my_prediction)
			return render_template('result1.html', prediction=my_prediction)



if __name__ == '__main__':
		app.run(debug=True)

