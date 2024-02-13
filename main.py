from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
import requests,openai,os
from dotenv.main import load_dotenv
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
llm = OpenAI(model="gpt-3.5-turbo-instruct")
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
app = Flask(__name__)
CORS(app)
from PyPDF2 import PdfReader



@app.route('/')
def index():
    return render_template('index.html')
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

pdf = PdfReader("Sigiriya.pdf")
pdf_reader = pdf

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Create the knowledge base object
knowledgeBase = process_text(text)



@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text=data.get('data')
    docs = knowledgeBase.similarity_search(text)
    user_input = text
    try:
        chain = load_qa_chain(llm, chain_type='stuff')
        output = chain.run(input_documents=docs,memory=memory, question=user_input)

        # conversation = ConversationChain(llm=llm,memory=memory,input_documents=docs)
        # output = conversation.predict(input=user_input)
        memory.save_context({"input": user_input}, {"output": output})
        return jsonify({"response":True,"message":output})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message":error_message,"response":False})
    
if __name__ == '__main__':
    app.run()
