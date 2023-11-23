import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

#config Site
st.set_page_config(page_title='Gottfried',
                   page_icon=':zany_face:',
                   layout='wide')
#Header
st.header("📄 Chat with the Incredible Gottfried 🤖")
#Sidebar
st.sidebar.title("Options")

model_type = st.sidebar.radio("Model Type", ("7B", "13B"))
temperature = st.sidebar.slider("Creativity:", min_value=0.0, max_value=1.0,value=0.75, step = 0.01)

#Build Model
def build_llm(model_type, temperature):
    path = 'YOURMODELPATH'
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model_path = ''
    if model_type == '7B':
        model_path = f'{path}MODELNAME'
    elif model_type == '13B':
        model_path = f'{path}MODELNAME'
    llm = LlamaCpp(model_path = model_path, temperature = temperature,
                   max_tokens=2000,
                   top_p=1,
                   callback_manager=callback_manager,
                   verbose=True,
                   n_ctx=2048,
                   stop_token = ['Human: '],
                   stop = ['Human:', 'Please answer', "Assistant's Response:", "Assistant:"])
    return llm

def build_prompt_template():
    prompt_template = """You are a helpful assistant. Always answer every question as helpfully as possible.  
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
        You are constantly learning and improving.
        You can use the history.
        {history}
        Current Conversation:
        Human: {input}
        Assistant:"""
    prompt = PromptTemplate(template = prompt_template, input_variables = ["history", "input"])
    return prompt

#memory
def build_memory():
    memory = ConversationBufferMemory(ai_prefix = "Assistant", memory_key = 'history', return_messages = True)
    return memory

#load model
def load_model(model_type, temperature):
    llm = build_llm(model_type=model_type, temperature=temperature)
    prompt = build_prompt_template()
    conversation_chain = ConversationChain(prompt=prompt, llm=llm, memory = build_memory())
    return conversation_chain

#Delete History und Load new Model
if st.sidebar.button(label = 'Initialize Session', key = 1):
    st.session_state.messages = []
    st.session_state.model = load_model(model_type=model_type, temperature=temperature)

if "model" not in st.session_state:
    conversation_chain = load_model(model_type=model_type, temperature=temperature)
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
            response = st.session_state.model.run(input = question)
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
