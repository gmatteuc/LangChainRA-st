# import the necessary libraries
import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# get the environment variables
load_dotenv()

# define the directory where the database should be located
current_dir = os.path.abspath(os.getcwd())
persist_directory = os.path.join(os.path.dirname(current_dir), 'chroma_db')
# check if the directory exists
if os.path.exists(persist_directory):
    # load persistent vectorstore from disk if it exists
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
else:
    # create persistent vectorstore if the directory does not exist
    loader = PyPDFLoader("./documents_db/Matteucci_2024.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500) # important!
    paper_chunks = loader.load_and_split(text_splitter=splitter)
    vectorstore = Chroma.from_documents(
        paper_chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,       
    )

# define llm to be used
# llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0) 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) 

# define the retriever to be used
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}) # could also use "mmr"

# define the function to get the response
def get_response(input, chat_history):

    # define contextualization prompt for history-aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # define history-aware retriever chain
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # define qa prompt for question answering
    qa_system_prompt = """
        You are an assistant to Dr. Giulio Matteucci, a neuroscientist.
        Your role is to engage with visitors on his personal website, providing answers to their inquiries
        regarding Giulio's review paper you are given access to. Your responses should be scientifically precise,
        and accuratily reflect the content and message of the papers.
        It is of paramount importance not to make up any information, when in doubt,
        just say that this question goes beyond the scope of the paper. 
        It is really important to ensure factual accuracy and avoid inventing concepts references and attributions.
        Strive to be clear and accessible for all users, be both professional and approachable,
        ensuring your explanations are succinct and schematic without sacrificing essential details.
        When answering questions, rely solely on the context provided by Dr. Matteucci's papers.
        \n\n 
        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # define question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # define rag chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # make sure rag chain outputs answer only
    full_chain = rag_chain.pick("answer")

    return full_chain.stream({
        "chat_history": chat_history,
        "input": input,
    })

# initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a chatbot assistant of Dr. Giulio Matteucci. How can I help you?"),
    ]

# handle conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# handle user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history)) #st.write(get_response(user_query, st.session_state.chat_history)) #

    st.session_state.chat_history.append(AIMessage(content=response))