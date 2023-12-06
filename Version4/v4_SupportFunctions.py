import json
from datetime import datetime
from langchain.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict

# Build Model
def build_llm(temperature):
    path = 'YOURMODELPATH'
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    model_path = f'{path}YOURMODELNAME'
    llm = LlamaCpp(model_path = model_path, temperature = temperature,
                   max_tokens=2000,
                   top_p=1,
                   callback_manager=callback_manager,
                   verbose=True,
                   n_ctx=32768,
                   n_gpu_layers = 30,
                   n_batch = 512,
                   stop = ['Human:', 'Please answer', "Assistant's Response:", "Assistant:"])
    return llm


def build_prompt_template(default_template):
    if default_template is None:
        prompt_template = """You are a helpful assistant. Always answer every question as helpfully as possible.  
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
            You are constantly learning and improving.
            You can use the history.
            {history}
            Current Conversation:
            Human: {input}
            Assistant:"""
    else:
        prompt_template = f"{default_template}" + """
        You are constantly learning and improving.
        You can use the history.
        {history}
        Current Conversation:
        Human: {input}
        Assistant:"""
    prompt = PromptTemplate(template = prompt_template, input_variables = ["history", "input"])
    return prompt

# memory
def build_memory():
    memory = ConversationBufferMemory(ai_prefix = "Assistant", memory_key = 'history', return_messages = True)
    #memory = ConversationBufferWindowMemory(ai_prefix="Assistant", memory_key='history', return_messages=True, k = 12)
    return memory

# load model
def load_model(temperature, memory, prompt_template):
    llm = build_llm(temperature=temperature)
    prompt = build_prompt_template(default_template=prompt_template)
    conversation_chain = ConversationChain(prompt=prompt, llm=llm, memory=memory)
    return conversation_chain


# save chat
def save_chat(model):
    extracted_messages = model.memory.chat_memory.messages
    memory_dict = messages_to_dict(extracted_messages)
    json_history = json.dumps(memory_dict)
    return json_history


def date_time_now():
    date = datetime.now().strftime('%Y%m%d')
    time = datetime.now().strftime('%H%M%S')
    return date, time
