#!/usr/bin/env python

import os
import sys
import json

from flask import Flask, render_template, Response, jsonify, request
from threading import Thread, Condition
from json2html import *

class WebServer(Thread):
	def __init__(self, port=9500):
		Thread.__init__(self)
		self.app = Flask(__name__)
		self.port = port
		self.cv = Condition()
		self.document = None
		
	def run(self):
		print("Web Server thread started")
		app = self.app
		@app.route('/')
		def index():
			return render_template('index.html')

		@app.route('/data')
		def data():	
			return Response(self.dataHandler(), mimetype='text/event-stream')

		if __name__ == '__main__':
			logger.debug("startng the application")
		self.app.run(host='0.0.0.0', port=str(self.port), threaded=True)

	def dataHandler(self): 
		while self.is_alive():
			self.cv.acquire()
			while self.document is None:
				self.cv.wait()
			if self.document is not None:
				yield f"data:{json.dumps(self.document)}\n\n"
				self.document = None
			self.cv.release()

	def push(self, document):
		self.cv.acquire()
		self.document = json2html.convert(json = document)
		self.cv.notify()
		self.cv.release()
