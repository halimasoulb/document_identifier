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

		self.app.run(host='0.0.0.0', port=self.port, threaded=True)

	def dataHandler(self): 
		
		yield f"data:{json.dumps({'title': 'Please scan your document'})}\n\n"
		while self.is_alive():
			self.cv.acquire()
			while self.document is None:
				self.cv.wait()
			if self.document is not None:
				data = {}
				if 'type' in self.document:
					data['title'] = self.document['type']
					del self.document['type']
				data['body'] = json2html.convert(json = self.document, 
												 table_attributes="id=\"document-table\" class=\"table table-bordered\"")
 
				yield f"data:{json.dumps(data)}\n\n"
				self.document = None
			self.cv.release()

	def push(self, document):
		self.cv.acquire()
		self.document = document.copy()
		self.cv.notify()
		self.cv.release()
