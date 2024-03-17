def redactor_context(theme: str, information: str):
	context = f"""I would love your help in writing the best blog post 
possible. I'd like my article to be well written and provide
something useful or have an impact on the reader.
	
Here's the theme: {theme} And here are the first information
	
Here are the information already found: {information}"""

	return context

def get_basic_questions(theme: str):
	basic_questions = [
		f"What are the main concepts and their definition related to {theme}?",
		f"What are the named entities related to {theme}?",
	]
	return basic_questions