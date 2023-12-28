import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css,bot_template, user_template


from dotenv import load_dotenv
load_dotenv()



def save_uploaded_file(file):
    """Saves the uploaded file to the Streamlit session state."""
    with open(file.name, "wb") as f:
        f.write(file.read())
    st.session_state["uploaded_file"] = file

def text_loader(upload_file,main_placeEditor):
    st.sidebar.write("File Uploaded Successfully!")
    save_uploaded_file(upload_file)
    # elements = partition(upload_file.name)
    Loader = UnstructuredPDFLoader(upload_file.name)
    print("Loaded success")
    main_placeEditor.text("DATA Loading....... Started ✅✅")
    data = Loader.load()  # returns documents
    time.sleep(2)
    main_placeEditor.text("DATA Loading....... Completed ✅✅")
    return data

def url_text_loader(urls,main_placeEditor):
    Loader = UnstructuredURLLoader(urls=urls)
    main_placeEditor.text("DATA Loading....... Started ✅✅")
    data = Loader.load()  # returns documents
    time.sleep(2)
    main_placeEditor.text("DATA Loading....... Completed ✅✅")
    return data



def split_data(data,main_placeEditor):
    textSpliter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = textSpliter.split_documents(data)
    main_placeEditor.text("Document chunks.......✅✅")
    return docs

def embed_data(docs,main_placeEditor):
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    main_placeEditor.text("Document Embedded....... ✅✅")
    time.sleep(2)
    return vectorstore

def save_data(vectorstore):
    with open("faissfile.pkl", 'wb') as f:
        pickle.dump(vectorstore, f)


llm = OpenAI(temperature=0.9,max_tokens=500)

def page_one():
    st.title("Article Research Tool 💹")

    st.sidebar.title("Enter the Resources: ")


    urls = []
    url1 = st.sidebar.text_input("Enter the URL 1 : ")
    urls.append(url1)
    url2 = st.sidebar.text_input("Enter the URL 2: ")
    urls.append(url2)
    url3 = st.sidebar.text_input("Enter the URL 3 : ")
    urls.append(url3)
    process = st.sidebar.button("Process URL")

    main_placeEditor = st.empty()
    if process:
        # data loading
        data = url_text_loader(urls,main_placeEditor)
        # spliting the data into chuncks
        docs = split_data(data,main_placeEditor)
        # embedding
        vectorStore = embed_data(docs,main_placeEditor)
        # save Data
        save_data(vectorStore)

    query = main_placeEditor.text_input("Enter the Question: ")
    erase = st.button("Erase", type="secondary")
    if query:
        if os.path.exists("faissfile.pkl"):
            with open("faissfile.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)


    if erase:
        os.remove("faissfile.pkl")
        url = []
        query = ""


def page_two():
    st.title("PDF Research Tool 📑")

    main_placeEditor = st.empty()

    st.sidebar.title("Upload PDF")
    upload_file = st.sidebar.file_uploader("Upload the pdf")
    process = st.sidebar.button("process")
    if process:
        # data loading
        data = text_loader(upload_file,main_placeEditor)
        # spliting the data into chuncks
        docs = split_data(data,main_placeEditor)
        # embedding
        vectorStore = embed_data(docs,main_placeEditor)
        # save Data
        save_data(vectorStore)

    query = main_placeEditor.text_input("Enter the Question: ")
    erase = st.button("Erase", type="secondary")
    if query:
        if os.path.exists("faissfile.pkl"):
            with open("faissfile.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)

    if erase:
        os.remove("faissfile.pkl")
        os.remove(upload_file.name)
        upload_file = ""
        query = ""


# Create a sidebar with radio buttons for page selection
st.sidebar.title("Choose the file type you working with")
selected_page = st.sidebar.radio("", ("Articles", "Pdf"))
# Display content based on the selected page
if selected_page == "Articles":
    page_one()
else:
    page_two()


