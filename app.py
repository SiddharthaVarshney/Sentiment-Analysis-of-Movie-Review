
from flask import Flask, render_template, request
import nltk
import time
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from bs4 import BeautifulSoup
from urllib.request import urlopen,Request
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('punkt')
import pickle
from transformers import pipeline
# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename2 = 'movie-genre-mnb-model.pkl'
classifier2 = pickle.load(open(filename2, 'rb'))
cv2 = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__,template_folder='template')

def isa_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	summarizer_lsa = LsaSummarizer()
	summary_2 =summarizer_lsa(parser.document,3)
	summary_list = [str(sentence) for sentence in summary_2]
	result = ' '.join(summary_list)
	return result
@app.route('/')
def page():
    return render_template('page.html')

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/ind2')
def ind2():
    return render_template('ind2.html')
@app.route('/index3')
def index3():
	return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        sent_pipeline = pipeline("sentiment-analysis")
        a = sent_pipeline(data)
        b=a[0]
        b=b['label']
        print(b)

    return render_template('result.html', prediction=b)

@app.route('/predict2', methods=['POST'])
def predict2():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv2.transform(data).toarray()
        my_prediction = classifier2.predict(vect)
        return render_template('res2.html', prediction=my_prediction)

@app.route('/process',methods=['GET','POST'])
def process():
    start = time.time()
    if request.method == 'POST':
        input_text = request.form['input_text']
        final_summary= isa_summary(input_text)

    return render_template('result3.html',ctext=input_text,final_summary=final_summary)


def get_text(url):
    reqt = Request(url,headers={'User-Agent' : "Magic Browser"})
    page = urlopen(reqt)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

@app.route('/process_url',methods=['GET','POST'])
def process_url():
	start = time.time()
	if request.method == 'POST':
		input_url = request.form['input_url']
		raw_text = get_text(input_url)

		final_summary = isa_summary(raw_text)

	return render_template('result3.html',ctext=raw_text,
                        final_summary=final_summary,
                        )



if __name__ == '__main__':
    app.run(debug=True)



