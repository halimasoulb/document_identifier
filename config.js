[
	{
		"name":"CIN",
		"re":"^.*\\nroyaume du maroc\\n.*\\ncarte nationale d.*\\n.*\\n(?P<Prenom>\\w+.*)\\n(?P<Nom>\\w+.*)\\nne le (?P<Date_de_naissance>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4}).*\\n(?P<Lieu_de_Naissance>\\w+.*)\\nvalable jusqu[\\w]?au (?P<Date_de_validite>[0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4})\\n.*\\n.*\\n(?P<Numero_de_carte_nationale>\\w{1,2}\\d{6})"
	},
	{
		"name":"PASSEPORT",
		"re":"r'Passeport'"
	}

]
