import re
import json


cin_re=r"^.*\nroyaume du maroc\n.*\ncarte nationale d.*\n.*\n(?P<Prenom>\w+.*)\n(?P<Nom>\w+.*)\nne le (?P<Date_de_naissance>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4}).*\n(?P<Lieu_de_Naissance>\w+.*)\nvalable jusqu[\w]?au (?P<Date_de_validite>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4})\n.*\n.*\n(?P<Numero_de_carte_nationale>\w{1,2}\d{6})"

cin_text="lucalallas\nroyaume du maroc\nasiball agla\ncarte nationale dridentite\njuli jla\njamal idin\nel youssefi\nne le 3101 1969 slip\nlgharbia sidi bennour\nvalable jusquaau 15052020\ncilgall collection plejl\nme\nm220703"


document_identifier = re.compile(cin_re)

match = document_identifier.match(cin_text)

if match is not None:
	result = match.groupdict()
	print(json.dumps(result, indent=4))


