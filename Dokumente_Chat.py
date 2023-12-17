import streamlit as st
from streamlit_chat import message
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain, ConversationChain, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#config Site
st.set_page_config(page_title='HKI',
                   page_icon=':zany_face:',
                   layout='wide')

#Header
st.header("📄 Chat with the Incredible Document Gottfried 🤖")
#Sidebar
st.sidebar.title("Options")
temperature = st.sidebar.slider("Creativity:", min_value=0.0, max_value=1.0,value=0.01, step = 0.01)

#Load VectorStore
def load_vector_db():
    DB_FAISS_PATH = "docs/"
    #Load Embeddings
    embeddings = LlamaCppEmbeddings(model_path='YOUREMBEDDINGSPATH')
    vectordb = Chroma(persist_directory=DB_FAISS_PATH,embedding_function= embeddings)
    return vectordb

#Build Model
def build_llm(temperature):
    path = 'models/'
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model_path = f'{path}mistral-7b-instruct-v0.1.Q8_0.gguf'
    llm = LlamaCpp(model_path = model_path, temperature = temperature,
                   max_tokens=2000,
                   top_p=1,
                   callback_manager=callback_manager,
                   verbose=True,
                   repetition_penalty = 1.1,
                   n_ctx=2048,
                   stop_token=['Human: '],
                   stop=['Human', ':']
                   )
    return llm

def build_prompt_template():
    prompt_template = """Du bist ein Intelligenter, wütender KI-Assistent. Du nutzt den bereitgestellten Kontext um Fragen zu beantworten, bist aber immer aggressiv.
    Du antwortest immer Wahrheitsgemäß. Wenn du Sachen nicht weißt, antworte: "Das weiß ich nicht" 
    Kontext:
    {context}
    History:    
    {chat_history}
    Aktuelle Unterhaltung:
    Human: {question}
    Assistant:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["chat_history","context", "question"])
    return prompt

#Current conversation:

#memory
def build_memory():
    memory = ConversationBufferMemory(ai_prefix = "Assistant", memory_key = 'chat_history', input_key = 'question' ,return_messages = True)
    #memory = ConversationBufferWindowMemory(ai_prefix="Assistant", memory_key = "history", k=5)
    return memory

def build_retreiver():
    vectordb = load_vector_db()
    vectordb.get()
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    return retriever

#load model
def load_model(temperature):
    llm = build_llm(temperature=temperature)
    prompt = build_prompt_template()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever=build_retreiver(),memory = build_memory(),
                                                               combine_docs_chain_kwargs={'prompt': prompt})
    return conversation_chain

#Delete History und Load new Model
if st.sidebar.button(label = 'Clear History', key = 1):
    st.session_state.messages = []
    st.session_state.model = load_model(temperature=temperature)

if "model" not in st.session_state:
    conversation_chain = load_model(temperature=temperature)
    st.session_state['model'] = conversation_chain

#chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input('Input your prompt here')
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role":"user", "content": question})
    with st.chat_message("assistant"):
        with st.spinner("Generating"):
            response = st.session_state.model({'question': question})['answer']
            st.markdown(response)
    st.session_state.messages.append({"role":"assistant", "content": response})



# Style Options
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)
