import sys

def generate_question_templates():
	with open("prof_qa/data_all/templates.txt", "w") as f:
		f.write("OFFICE\tWhere can I meeet {person}?\tWhere is {person}'s office?\tWhere is {person}'s office located at?\tWhere in the academic building is {person}'s office located at?\tTell me where {person}'s office is.\nSCHOOL\tWhat's {person}'s academic background?\tWhich school did {person} graduate from?\tWhere did {person} get their degree from?\tWhat major did {person} graduate with?\tWhat university did {person} go to?\nINTEREST\tWhat research is {person} currently working on?\tWhat is {person}'s research interest?\tWhat field of research does {person} specialize in?\tWhat is {person}'s interest as an academia?\tWhich field of research does {person} focus on?\nBIO\tTell me about {person}.\tWhat is the bio of {person}?\tWhat is the biography of {person}?\tWho is {person}?\tGive me a brief description of {person}.\nCOURSES\tWhat is {person} teaching?\tWhat courses does {person} teach?\tWhat courses are {person} currently teaching?\tWhat courses has {person} taught?\tShow me a list of courses {person} teaches.\nPIC\tHelp me recognize {person}.\tShow me a picture of {person}.\tWhat does {person} look like?\tHow would I recognize {person} if I met them?\tI need a photo of {person}.\nEMAIL\tI'd like to email {person}.\tWhat is email address of {person}?\tWhat's {person}'s email?\tHow can I contact {person} via email?\tI'd like to write an email to {person}.\nPHONE\tI'd like to call {person}.\tWhat is the phone number of {person}?\tHow can I contact {person} via phone?\t I'd like to contact {person} by phone call.\tWhat's {person}'s phone number?\n")

if __name__ == "__main__":
	generate_question_templates()
	
