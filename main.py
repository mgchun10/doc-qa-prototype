import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing the Key")

loader = PyMuPDFLoader("NBADraftTipSheet.pdf")
documents = loader.load()
print(f"{len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
print(f"{len(docs)} chunks.\n")

# TEST: Print few links
# for i, chunk in enumerate(docs):
#     print(f"------- Chunk {i} -------")
#     lines = chunk.page_content.split("\n")
#     for line in lines[:5]:
#         print(line)
#     print()

# TEST
# print("----- TEST ------")
# print(docs[0].page_content[:300] + "...\n")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")  

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

question = "Who is eligible for the NBA draft?"
answer = qa_chain.run(question)
print(f"\nQ: {question}\nA: {answer}")