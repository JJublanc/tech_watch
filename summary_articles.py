from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from prompts import redactor_context, get_basic_questions
from googlesearch import search
import os
import dotenv

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
redactor_instruction_question = os.getenv("REDACTOR_CONTEXT_QUESTION")
redactor_instruction_redaction = os.getenv("REDACTOR_CONTEXT_REDACTION")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106",
                 openai_api_key=openai_api_key)


def get_url_from_google(query: str):
	# Créez une liste pour stocker les URL
	urls = []

	for j in search(query, num_results=3):
		# Ajoutez chaque URL à la liste
		urls.append(j)
	return urls

def rag(llm, embeddings, docs, query):
	prompt = ChatPromptTemplate.from_template(
		"""Answer the following question based only on the provided context:
		
		<context>
		{context}
		</context>
		
		Question: {input}""")
	document_chain = create_stuff_documents_chain(llm, prompt)

	text_splitter = RecursiveCharacterTextSplitter()
	documents = text_splitter.split_documents(docs)

	vector = FAISS.from_documents(documents, embeddings)
	retriever = vector.as_retriever()
	retrieval_chain = create_retrieval_chain(retriever, document_chain)
	response = retrieval_chain.invoke(
		{"input": query})

	return response


def redactor(llm, context, query):
	prompt = ChatPromptTemplate.from_messages([
		("system", context),
		("user", "{input}")
	])

	output_parser = StrOutputParser()

	chain = prompt | llm | output_parser

	return chain.invoke({"input": query})


def keep_going():
	keep_going_value = input(
		"\n\n Would you like to ask another question? (yes/no) \n\n")
	print(keep_going_value=="yes")
	if keep_going_value== "no":
		return False
	elif keep_going_value == "yes":
		return True
	else:
		return keep_going()


if __name__ == "__main__":
	keep_going_value = True
	theme = "vector search engine comparison"
	information = ""
	article= {"answer":""}
	urls = get_url_from_google(theme)

	loader = WebBaseLoader(urls)
	docs = loader.load()
	basic_questions = get_basic_questions(theme)
	asked_questions = basic_questions

	for question in basic_questions:
		information += rag(llm, embeddings, docs, question)["answer"] + "\n\n"


	while keep_going_value == True:

		#############################
		# 1. Redacteur ask question #
		#############################
		context = redactor_context(theme, information)
		print("\n\nCONTEXT\n--------------------\n" + context + "\n\n")
		print("\n\nINSTRUCTION\n--------------------\n" + redactor_instruction_question + "\n\n")
		redactor_instruction_question += f"\n\nAsk another question than the following: {asked_questions}"
		redactor_question = redactor(llm, context, redactor_instruction_question)
		asked_questions.append(redactor_question)

		##########################
		# 2. RAG answer question #
		##########################
		print(
			"\n\nREDACTOR question \n--------------------\n"
			+ redactor_question + "\n\n")
		information += rag(llm, embeddings, docs, redactor_question)["answer"] + "\n\n"
		print("\n\nINFO extracted \n--------------------\n" + information + "\n\n")

		##################################
		# 3. Redactor redact the article #
		##################################
		keep_going_value = keep_going()
		if not keep_going_value:
			context = redactor_context(theme, information)
			print("\n\nCONTEXT\n--------------------\n" + context + "\n\n")
			print("\n\nINFORMATION\n--------------------\n" + information + "\n\n")
			article = redactor(llm, context, redactor_instruction_redaction)

			#########################
			# 4. Format the article #
			#########################
			context = f"Here is an article about {theme}:\n\n{article}"
			query = "format the article using markdown"
			article = redactor(llm, context, query) + "\n\n" + f"Sources: {urls}"

	with open(f'article_{"_".join(theme.split(" "))}.md', 'w') as file:
		file.write(article)
