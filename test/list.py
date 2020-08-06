#mylist=[4,1,10,5,30]
#mylist=mylist.sort()
#print(mylist)
import re
import datetime

text="lucalallas\nroyaume du maroc\nasiball agla\ncarte nationale dridentite\njuli jla\njamal idin\nel youssefi\nne le 3101 1969 slip\nlgharbia sidi bennour\nvalable jusquaau 15052020\ncilgall collection plejl\nme\nm220703"
result=re.match(r"^.*\nroyaume du maroc\n.*\ncarte nationale d.*\n.*\n(?P<Prenom>\w+.*)\n(?P<Nom>\w+.*)\n(?P<Date_de_naissance>ne le [0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4}).*\n(?P<Lieu_de_Naissance>\w+.*)\n(?P<Date_de_validite>valable jusq.* [0-9]{2}[ ,1]?[0-9]{2}[ ,1]?[0-9]{4})\n.*\n.*\n(?P<Numero_de_carte_nationale>\w{1,2}\d{6})", text)
print("information carte nationale:", result.group(1,2,3,4,5,6))


#madate='30082020'
#date_time_object=datetime.datetime.strptime(madate, '%d%m%Y')
#print(' Date de naissance:', date_time_object.date())



#list=sorted([4,1,10,5,30])
#print(list)
#numbers = (1, 2, 3, 4) 
#result = map(lambda x: x + x, numbers) 
#print(list(result))

#list1=[1,2,3,4,5]
#list2=['a','b','c','d','e']
#list3=map(lambda index:list1.index(1),list2)
#print(list(list3))

#result=re.search(r"abc", "abcfdr")
#print(result)

#chn_mdp=r"^[A-Za-z0-9]{6,}$"
#exp_mdp=re.compile(chn_mdp)

#s="La vie est belle"

#replace={'vie':'halima', 'belle':'jolie'}

#reduce(lambda a,kv:a.replace(*kv), replace, s)



#mot_de_passe=""
#while exp_mdp.search(mot_de_passe) is None:
	#mot_de_passe=input("Entrer votre mot de passe :")

