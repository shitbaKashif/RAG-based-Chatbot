# from app import Flask, render_template, request, jsonify
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import CommaSeparatedListOutputParser

app = Flask(__name__)
# Define your model setup code here
def model_load():
    local_path = "books\Fundamentals-of-Psychological-Disorders.pdf"
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag"
    )

    local_model = "phi3"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain,vector_db
# Define route for frontend
chain,vector_db = model_load()

@app.route("/model", methods=["GET"])
def home():
    chain,vector_db = model_load()
    return render_template("chat.html")
@app.route("/", methods=["GET"])
def chat():
    return render_template("chat.html")

# Define route for handling queries
@app.route("/query", methods=["POST"])
def query_handler():
    print(request.form)
    query = str(request.form["query"])
    output = chain.invoke(query)
    return jsonify({"output": output})

@app.route("/rem", methods=["GET"])
def remove():
    return vector_db.delete_collection()

if __name__ == "__main__":
    app.run(debug=True)
#mkaing variable global
