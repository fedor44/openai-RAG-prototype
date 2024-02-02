__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
import tempfile
import os
import bs4

# load env
load_dotenv()

# method for uploading file
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# title
st.title("ChatPDF")
st.write("---")

# upload file
uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf'])
st.write("---")

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # embedding with openai
    embeddings = OpenAIEmbeddings()

    # save data to Chroma
    vectorstore = Chroma.from_documents(texts, embeddings)

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("http://wiki.hash.kr/index.php/%EA%B9%80%ED%98%95%EC%A7%84",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    splits = text_splitter.split_documents(docs)

    # make a question
    st.header("PDF에게 질문하세요.")
    question = st.text_input("질문을 입력 하세요")

    if st.button("질문하기"):
        with st.spinner("wait for it..."):
            # make a question
            #question = "세종텔레콤 회장님 학력 좀 알려줘"
            #llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)
            llm = ChatOpenAI(model_name = "gpt-4-0125-preview", temperature = 0)

            # Retrieve and generate using the relevant snippets of the blog.
            retriever = vectorstore.as_retriever()
            prompt = hub.pull("rlm/rag-prompt")
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )

            result = rag_chain.invoke(question)
            st.write(result)
            #print(result)

#result = llm.invoke(question)
#print(result)