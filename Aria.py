import streamlit as st
from langchain import *
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory


load_dotenv()
import nltk
nltk.download('punkt')


os.environ['HF_KEY'] = os.getenv("HF_KEY")
repo_id = 'HuggingFaceH4/zephyr-7b-beta'  
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HF_KEY'],
                     repo_id=repo_id)


# Initialize language model and conversation memory
template = """You're name is Aria you are chatbot having a conversation with a human to his mental status so we can use music therapy to help him.
write one sentence as answer per time
{history}
You: {input}
Aria:
"""

prompt = PromptTemplate(
    input_variables=['history', 'input'],
    template=template
)

memory = ConversationBufferMemory(memory_key="history")
doc_chain = ConversationChain(llm=llm,prompt=prompt,
    memory=memory)


# Title and sidebar
st.title('Clinical Chatbot')
st.markdown('### Conversation')


# Initialize or retrieve session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'questions' not in st.session_state:
    st.session_state.questions = []

# User input
user_input = st.text_input('You:', key='input')

# Submit button
if st.button('Submit'):
    try:
        chat_history_text = '\n'.join(st.session_state.chat_history)
    except:
        chat_history_text=""
 

    st.session_state.chat_history.append(f'You: {user_input}')
    # Generate chatbot response
    response = doc_chain.predict(input=user_input)
    print(response)

    st.session_state.chat_history.append(f'Aria: {response}')
    # Clear user input
    user_input = ""


# Display chat history (not editable)
chat_history_text = '\n'.join(st.session_state.chat_history)
st.text_area('Chat History', value=chat_history_text, height=200, key='output')

# Custom CSS to improve contrast for the disabled text
custom_css = """
<style>
    .st-bj {
        background-color: #1f2937;
        color: #ffffff;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
