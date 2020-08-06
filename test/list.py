import re
import json

cin_text="lucalallas\nroyaume du maroc\nasiball agla\ncarte nationale dridentite\njuli jla\njamal idin\nel youssefi\nne le 3101 1969 slip\nlgharbia sidi bennour\nvalable jusquaau 15052020\ncilgall collection plejl\nme\nm220703"



with open('config.js') as json_file:
	documents = json.load(json_file)
	for doc in documents:
		
		doc_id = re.compile(str(doc['re']))
		match = doc_id.match(cin_text)
		if match is not None:
			result = match.groupdict()
			result['Document name'] = doc['name']
			for key, value in result.items():
				result[key] = value.upper()
			print(json.dumps(result, indent=4))

