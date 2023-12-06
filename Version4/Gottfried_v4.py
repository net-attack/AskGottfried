import json
import streamlit as st
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import messages_from_dict
import v4_SupportFunctions as sf

from PIL import Image
import numpy as np

# config Site
st.set_page_config(page_title='Gottfried',
                   page_icon=':zany_face:',
                   layout='wide')
# Header
st.header("📄 Chat with the Incredible Gottfried 🤖")
#Logo
logo = Image.open('YOURLOGOPATH')
st.sidebar.image(logo)

# Sidebar
st.sidebar.title("Options")


temperature = st.sidebar.slider("Creativity:", min_value=0.0, max_value=1.0,value=0.75, step = 0.01)

with st.sidebar.expander(label="Expert Modus Only"):
    default_template = st.text_area(label="Set individual Prompt template")

# Delete History und Load new Model
if "model" not in st.session_state:
    conversation_chain = sf.load_model(temperature=temperature, memory=sf.build_memory(), prompt_template= default_template)
    st.session_state['model'] = conversation_chain

col1, col2 = st.sidebar.columns(2)
with col1:
    st.download_button(
        label = "Download Chat",
        file_name=f"chat_history_{sf.date_time_now()[0]}_{sf.date_time_now()[1]}.json",
        mime= "application/json",
        data=sf.save_chat(st.session_state.model))
with col2:
    if st.button(label = 'Initialize Session', key = 1):
        st.session_state.messages = []
        st.session_state.model = sf.load_model(temperature=temperature, memory=sf.build_memory(), prompt_template= default_template)

if "file_name" not in st.session_state:
    st.session_state["file_name"] = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


upload_file = st.sidebar.file_uploader("Upload Chat History",type = ["json"])

if upload_file is not None:
    if upload_file.name != st.session_state["file_name"]:
        history_json = json.load(upload_file)
        retrieved_messages = messages_from_dict(history_json)
        chat_history = ChatMessageHistory(messages=retrieved_messages)
        history_memory = ConversationBufferMemory(chat_memory=chat_history)
        st.session_state.model = sf.load_model(temperature=temperature, memory=history_memory, prompt_template= default_template)
        st.session_state.messages = []
        for i in history_json:
            if i['data']['type'] == 'human':
                st.session_state.messages.append({"role":"user", "content": i['data']['content']})
            elif i['data']['type'] == 'ai':
                st.session_state.messages.append({"role":"assistant", "content": i['data']['content']})
        st.session_state["file_name"] = upload_file.name

# chat history
avatar = Image.open("YOURAVATARPATH")

for message in st.session_state.messages:
    if message["role"] == 'user':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"], avatar = np.array(avatar)):
            st.markdown(message["content"])

question = st.chat_input('Input your prompt here')
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=np.array(avatar)):
        with st.spinner("Generating"):
            response = st.session_state.model.run(input = question)
            st.markdown(response)
    st.session_state.messages.append({"role": "user", "content": question})
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
