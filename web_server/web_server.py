#!/usr/bin/env python

import os
import sys


from flask import Flask, render_template, Response, jsonify, request
#import threading

class WebServer():
	def __init__(self, port=9500):
		self.app = Flask(__name__)
		self.port = port
		

	def handler(self):
		app = self.app
		#threading.Thread(target=index).start()
		@app.route('/')
		def index():
			return render_template('index.html')
		if __name__ == '__main__':
			logger.debug("startng the application")
		self.app.run(host='0.0.0.0', port=str(self.port), threaded=True)