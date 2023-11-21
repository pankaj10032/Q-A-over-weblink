import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os

url="https://www.cranberry.fit/post/ovulation-pain-unmasking-the-mystery"

# Streamlit UI
st.title("Question Answering App")
def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text
def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=.7)
    return llm
os.environ['OPENAI_API_KEY']=get_api_key()
def get_text():
    input_text = st.text_area(label="Text Input", label_visibility='collapsed', placeholder="Your text...", key="yext_input")
    return input_text
text_input = get_text()
if text_input:
    if not os.environ['OPENAI_API_KEY']:
        st.warning('Please insert OpenAI API Key. Instructions [here]', icon="⚠️")
        st.stop()

    llm = load_LLM()

    loader = WebBaseLoader(url)
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    splitted_text = text_splitter.split_documents(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splitted_text, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []

    # query = st.text_input("Ask a question:")
    if st.button("Submit"):
        result = chain({"question": text_input, "chat_history": chat_history})
        st.write("Answer:", result['answer'])


