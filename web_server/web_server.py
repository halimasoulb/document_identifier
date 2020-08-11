#!/usr/bin/env python

import os
import sys

from flask import Flask, render_template, Response, jsonify, request
from threading import Thread

class WebServer(Thread):
	def __init__(self, port=9500):
		Thread.__init__(self)
		self.app = Flask(__name__)
		self.port = port
		

	def run(self):
		print("Web Server thread started")
		app = self.app
		@app.route('/')
		def index():
			return render_template('index.html')
		if __name__ == '__main__':
			logger.debug("startng the application")
		self.app.run(host='0.0.0.0', port=self.port, threaded=True)