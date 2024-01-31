from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# load pdf file
loader = PyPDFLoader("sejong.pdf")
pages = loader.load_and_split()

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

# make a question
question = "세종텔레콤 회장님 학력 좀 알려줘"
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)
#llm = ChatOpenAI(model_name = "gpt-4-1106-preview", temperature = 0)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

result = rag_chain.invoke(question)
print(result)

#result = llm.invoke(question)
#print(result)