import os
import datetime
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template, session
import json
import tiktoken
import re
import cohere
import pickle
import numpy as np
import time
from langchain_community.chat_models import ChatOpenAI

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this_is_bad_secret_key'

# get your openai api key at https://platform.openai.com/
OPENAI_API_KEY = 'your_openai_api_key'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# get your cohere api key at https://cohere.com/
COHERE_KEY = 'your_cohere_api_key'
co = cohere.Client(COHERE_KEY)

limit_input_tokens=4096
file_path = 'data/dataset_folder/dataset_hsbc'
prompt_template_path = 'data/prompt_template/prompt_template.txt'
num_chunks_of_text = 8 # number of chunks of text to search via embeddings similarity


def read_dataset():
	# read file
	with open(file_path, 'r', encoding="utf-8") as f:
		document = f.read()
	document_chunks = document.split('\n\n\n')
	return document_chunks

document_chunks = read_dataset()
print('doc_chunks len: ', len(document_chunks))

def read_template():
	# read file
	with open(prompt_template_path, 'r', encoding="utf-8") as f:
		prompt_template = f.read()
	return prompt_template

prompt_template = read_template()


def get_context_embeddings_co():
	context_emb = co.embed(document_chunks, input_type="search_document", model="embed-multilingual-v3.0").embeddings
	context_emb = np.asarray(context_emb)
	return context_emb

def generate_full_llm_query(query, document_chunks, prompt_template, limit_input_tokens=4096):

	# cohere embeddings
	query_emb = co.embed([query], input_type="search_query", model="embed-multilingual-v3.0").embeddings
	query_emb = np.asarray(query_emb)
	context_emb = get_context_embeddings_co()

	#Compute the dot product between query embedding and document embedding
	scores = np.dot(query_emb, context_emb.T).squeeze()

	max_idx = np.argsort(-scores)

	context_chunks_initial = []
	context_scores = []
	for idx in max_idx[:num_chunks_of_text]:
	  context_scores.append(scores[idx])
	  context_chunks_initial.append(document_chunks[idx])
	context_chunks = context_chunks_initial


	llm_full_query = ''

	# More complex logic is implemented to ensure the full llm query does not exceeds input token limit
	# i.e. finall llm query is cooked with size < limit_input_tokens
	correction_num_of_tokens = 500 # additionally decrease num of tokens by this number
	# Truncating chunks to the size of limit_input_tokens
	num_of_chunks = len(context_chunks)
	for iter in range(num_of_chunks):
		print('context_chunks len:', len(context_chunks))
		context_chunks_as_str = '\n###\n'.join([str(elem) for elem in context_chunks])
		llm_full_query = prompt_template.format(context=context_chunks_as_str, question=query)
		encoding = tiktoken.get_encoding("cl100k_base")
		num_tokens_template = len(encoding.encode(prompt_template))
		num_tokens = len(encoding.encode(llm_full_query))
		print('num_tokens', num_tokens)

		if num_tokens <= limit_input_tokens:
		  llm_full_query, num_tokens
		elif len(context_chunks) == 1 and num_tokens > limit_input_tokens - correction_num_of_tokens:
		  chunk_appendix = '\nDetails in the link:'
		  extracted_link = re.search(r'https://.+', context_chunks[0][-100:]).group(0)
		  chunk_appendix = chunk_appendix + ' ' + extracted_link
		  num_of_chars_to_cut = num_tokens - limit_input_tokens + num_tokens_template + correction_num_of_tokens
		  context_chunks[0] = context_chunks[0][:-num_of_chars_to_cut]
		  context_chunks[0] = context_chunks[0] + chunk_appendix
		  context_chunks_as_str = '\n###\n'.join([str(elem) for elem in context_chunks])
		  llm_full_query = prompt_template.format(context=context_chunks_as_str, question=query)
		  num_tokens = len(encoding.encode(llm_full_query))
		  llm_full_query, num_tokens
		elif num_tokens > limit_input_tokens - correction_num_of_tokens:
		  context_chunks = context_chunks[:-1]

	return llm_full_query, context_chunks


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
	data = request.get_json()
	if isinstance(data, str):
		data = json.loads(data)
	
	query = data['message']

	# Process user message here if needed
	now = datetime.datetime.now()
	llm_full_query, context_chunks = generate_full_llm_query(query, document_chunks, prompt_template, limit_input_tokens=limit_input_tokens)
	print('current full query:', llm_full_query)
	llm_answer = llm.invoke(llm_full_query)
	#llm_answer = llm_full_query  # this is a stub if llm initialization disabled
	llm_answer = llm_answer.content.strip()

	after = datetime.datetime.now()
	delta_time = after - now
	delta_time = round(delta_time.total_seconds())
	llm_answer = llm_answer + '\n\n' + '> response generation time: ' + str(delta_time) + ' seconds.'
	print(llm_answer)
	response = {'message': llm_answer}
	return jsonify(response)


if __name__ == "__main__":
	llm = ChatOpenAI(model_name='gpt-3.5-turbo')
	app.run(host='0.0.0.0', port=80)