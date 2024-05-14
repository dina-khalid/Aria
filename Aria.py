import streamlit as st
from langchain import *
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from Music_genrate import generate_music


load_dotenv()
# ---------------------- Elements styling -------------------------------- #

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



os.environ['HF_KEY'] = os.getenv("HF_KEY")
print(os.environ['HF_KEY'])
import huggingface_hub
print(huggingface_hub.__version__)
repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'  
llm = HuggingFaceEndpoint(huggingfacehub_api_token=os.environ['HF_KEY'],
                     repo_id=repo_id)


# Initialize language model and conversation memory
template = """You're name is Aria you are chatbot having a conversation with a human to his mental status so we can use music therapy to help him.
reply with one sentence only
{history}
Human: {input}
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
st.title('ARIA')



# Initialize or retrieve session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'questions' not in st.session_state:
    st.session_state.questions = []





# Display chat history (not editable)
chat_history_text = '\n'.join(st.session_state.chat_history)
st.text_area('Chat History', value=chat_history_text, height=200, key='output')
# Submit button
inputcol , buttoncol = st.columns([7,1])
with inputcol:
    # User input
    user_input = st.text_input('You:', key='input')
with buttoncol:
    if st.button('submit'):
        try:
            chat_history_text = '\n'.join(st.session_state.chat_history)
        except:
            chat_history_text=""
    

        st.session_state.chat_history.append(f'You: {user_input}')
        # Generate chatbot response
        response = doc_chain.predict(input=user_input)

        st.session_state.chat_history.append(f'Aria: {response}')
        


        # Clear user input
        user_input = ""
        # re-render 
        st.experimental_rerun()

def display_and_play_audio(filename="./output_audio.wav", mime_type="audio/wav"):
    wav_file = open(filename, "rb")
    if wav_file is not None:
        st.audio(wav_file.read())


if st.button('generate music'):
    print(st.session_state.chat_history[-1])
    sentences = st.session_state.chat_history[-1].split(".!?”")
    print(sentences[0][6:])
    audio_bytes =  generate_music(sentences[0][6:])
    if audio_bytes is not None:  # Check if audio generation was successful
      display_and_play_audio()
    else:
      st.error("Error generating audio. Please check your `generate_music` function.")

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
