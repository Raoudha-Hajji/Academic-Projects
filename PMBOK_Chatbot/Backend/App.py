from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
#from clean_text import clean_query, escape_regex_chars, format_output
import os
from dotenv import load_dotenv
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import base64
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize embeddings and environment variables
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Neo4j vector store
neo4j_graph_vector_index = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=url,
    username=username,
    password=password,
    node_label="Entity",
    index_name="entity",
    embedding_node_property="embedding",
    text_node_properties=["name", "definition"]
)

# Load documents
loader = TextLoader(r"C:/Users/RAOUDHA/Desktop/pmbok/Backend/extracted_text_with_figures.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize vector store
db = Neo4jVector.from_documents(docs, embedding_model, url=url, username=username, password=password)

# Initialize language model
llm = ChatGroq(model="llama-3.2-90b-text-preview")


figures_txt_path = r"C:/Users/RAOUDHA/Desktop/pmbok/Backend/List_Figures_Tables.txt"
figures_folder= r"C:/Users/RAOUDHA/Desktop/pmbok/Backend/extracted_figures"

# Initialize the sentence transformer model
figure_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the system prompt and human prompt templates
prompt = """
Your Job is to use the provided retriever data to answer
questions about the Project Management PMBOK.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's
not from the context. If you don't know an answer, say you don't know.

Important rules:
- Be as detailed as possible but DO NOT make up any information that is not from the context.
- If the context does not contain enough information to answer the question, say "I don't know" or "I cannot find this information in the provided context".
- Avoid answering with irrelevant or off-topic information. Stick strictly to the domain of Project Management PMBOK.
- Ensure that all responses relate solely to PMBOK concepts, processes, knowledge areas, or methodologies.

{context}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=prompt,
        input_variables=["context"]
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="Can you provide details on: {question}",
        input_variables=["question"]
    )
)

messages = [system_prompt, human_prompt]

qa_prompt = ChatPromptTemplate(
    messages=messages,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain with the LLM
qa_chain_text = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)

qa_chain_text.combine_documents_chain.llm_chain.prompt = qa_prompt

### ... Memory chat from here ... 

# Define prompts
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Your Job is to use the provided retriever data to answer
questions about the Project management PMBOK.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's
not from the context. If you don't know an answer, say you don't know.
{context}

Question: {question}
""")

# Initialize memory and chain components
memory = ConversationBufferMemory(
    return_messages=True,
    output_key="answer",
    input_key="question"
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Build the conversation chain
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | (llm | (lambda message: message.content))
}

retrieved_documents = {
    "docs": itemgetter("standalone_question") | neo4j_graph_vector_index.as_retriever(k=10),
    "question": lambda x: x["standalone_question"],
}

final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

answer = {
    "answer": final_inputs | ANSWER_PROMPT | (llm | (lambda message: message.content)),
    "question": itemgetter("question"),
    "context": final_inputs["context"]
}

final_chain = loaded_memory | standalone_question | retrieved_documents | answer


# Load figure titles from the text file
def load_figures_from_txt(file_path):
    with open(file_path, "r") as f:
        figures_list = f.readlines()
    figures_list = [figure.strip() for figure in figures_list]  # Clean up whitespace
    return figures_list


figures_list = load_figures_from_txt(figures_txt_path)



# Compute embeddings for both the query and figure titles
def compute_embeddings(text_list):
    return figure_model.encode(text_list)

# Encode query and figure titles
def compute_similarity(query_embedding, figure_embeddings):
    return cosine_similarity([query_embedding], figure_embeddings)[0]

# Rank figures based on similarity scores
def rank_figures_by_similarity(figures_list, similarity_scores, top_k=2):
    ranked_results = []
    for i, score in enumerate(similarity_scores):
        ranked_results.append((figures_list[i], score))
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]

# Avoid processing duplicate figures
def remove_duplicates(results):
    seen = set()
    unique_results = []
    for result in results:
        if result[0] not in seen:
            unique_results.append(result)
            seen.add(result[0])
    return unique_results

# Find and display relevant figures with image previews
def display_relevant_figures(figures_folder, ranked_figures):
    all_figure_files = os.listdir(figures_folder)
    figure_data = []

    for figure_title, score in ranked_figures:
        matched_files = [file for file in all_figure_files if figure_title in file]
        
        if matched_files:
            figure_path = os.path.join(figures_folder, matched_files[0])
            with open(figure_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            figure_data.append({
                "figure_title": figure_title,
                "image_base64": image_base64
            })

    return figure_data

def check_context_relevance(response):
    """
    Check if the response indicates no relevant information was found in the context
    """
    no_info_phrases = [
        "i don't know",
        "i don't have enough information",
        "i cannot find",
        "no information",
        "cannot answer",
        "don't have sufficient context",
        "not found in the context",
        "not mentioned in the context",
        "does not include",
        "unfortunately",
        "does not mention",
        "i don't have any information"
    ]
    
    response_lower = response.lower()
    return not any(phrase in response_lower for phrase in no_info_phrases)

# API endpoint for simple text response chat

@app.route('/simple-text-response', methods=['POST'])
def simple_text_response():
    query = request.json.get('query')
    
    # Get response from QA chain
    response = qa_chain_text.invoke({"query": query})
    result = response.get('result', '')
    
    # Check if the response indicates relevant context was found
    has_relevant_context = check_context_relevance(result)
    
    return jsonify({
        'query': query,
        'result': result,
        'has_relevant_context': has_relevant_context
    })

# API endpoint for chatbot with memory
@app.route('/chatbot', methods=['POST'])
def chatbot():
    query = request.json.get('query')
    
    # Use the conversational chain to get response
    result = final_chain.invoke({"question": query})
    
    # Save the context
    memory.save_context({"question": query}, {"answer": result["answer"]})
    
    # Return response in the expected format
    response = {
        "answer": result["answer"],
        "question": result["question"],
        "context": result["context"]
    }
    
    return jsonify(response)
@app.route('/figures', methods=['POST'])
def figures():
    data = request.get_json()
    query = data.get('query')
    has_relevant_context = data.get('has_relevant_context', False)

    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400
        
    # Only process figures if we have relevant context
    if not has_relevant_context:
        return jsonify([])  # Return empty array if no relevant context

    # Encode the query
    query_embedding = figure_model.encode([query])[0]
    figure_embeddings = compute_embeddings(figures_list)
    similarity_scores = compute_similarity(query_embedding, figure_embeddings)

    # Only include figures with similarity above threshold
    threshold = 0.3  # Adjust this threshold as needed
    ranked_figures = []
    for i, score in enumerate(similarity_scores):
        if score > threshold:
            ranked_figures.append((figures_list[i], score))
    
    ranked_figures.sort(key=lambda x: x[1], reverse=True)
    ranked_figures = ranked_figures[:2]  # Keep top 2 figures
    ranked_figures = remove_duplicates(ranked_figures)

    figure_data = display_relevant_figures(figures_folder, ranked_figures)
    
    return jsonify(figure_data)
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, port=5000)