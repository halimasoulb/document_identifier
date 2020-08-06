import re
import json


cin_re2=r'^.*\nroyaume du maroc\n.*\ncarte nationale d.*\n.*\n(?P<Prenom>\w+.*)\n(?P<Nom>\w+.*)\nne le (?P<Date_de_naissance>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4}).*\n(?P<Lieu_de_Naissance>\w+.*)\nvalable jusqu[\w]?au (?P<Date_de_validite>(?P<day>[0-9]{2})[ ,1]?(?P<month>[0-9]{2})[ ,1]?(?P<year>[0-9]{4}))\n.*\n.*\n(?P<Numero_de_carte_nationale>\w{1,2}\d{6})'

cin_re=r'^.*\nroyaume du maroc\n.*\ncarte nationale d.*\n.*\n(?P<Prenom>\w+.*)\n(?P<Nom>\w+.*)\nne le (?P<Date_de_naissance>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4}).*\n(?P<Lieu_de_Naissance>\w+.*)\nvalable jusqu[\w]?au (?P<Date_de_validite>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4})\n.*\n.*\n(?P<Numero_de_carte_nationale>\w{1,2}\d{6})'

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

