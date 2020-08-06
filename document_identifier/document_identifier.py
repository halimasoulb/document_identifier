import re
import json

import sys
import os



class DocumentIdentifier():
	def __init__(self, config_file):
		with open(config_file) as json_file:
			self.documents = json.load(json_file)

		

	def process(self, text):
		for doc in self.documents:
			doc_id = re.compile(str(doc['re']))
			match = doc_id.match(text)
			if match is not None:
				result = match.groupdict()
				result['Document name'] = doc['name']
				for key, value in result.items():
					result[key] = value.upper()
				else:
					return None
				return result

    
    	
