#!/usr/bin/python
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def hello():
	print request.form['srcTxt'];
	return render_template('index.html')


if __name__ == '__main__':
	app.run()