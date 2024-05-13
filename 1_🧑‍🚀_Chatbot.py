import streamlit as st
from llm_chains import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
# from image_handler import handle_image
from pdf_handler import add_documents_to_db
from audio_handler import transcribe_audio
import os
import yaml
import time

from PIL import Image

img = Image.open('./backgrounds/04d2fc00325f33c20b55451a7618d02a.jpg')

st.set_page_config(page_title='', page_icon=img)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input 
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config['chat_history_path'] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config['chat_history_path'] + st.session_state.session_key)

def main():
    st.title('Local Educational Chatbot')
    chat_container = st.container()
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config['chat_history_path'])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key = 'session_key', index = index, on_change = track_index)
    
    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config['chat_history_path'] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key = "history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Type your message here", key = "user_input", on_change = set_send_input)

    col1, col2, col3, col4, col5 = st.columns([0.51, 0.1, 0.6, 0.5, 1])

    with col1:
        voice_recording = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", just_once=True)
    with col4:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        start_time = time.time()
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)
        end_time = time.time()
        print(f"Time taken for LLM for voice: {end_time - start_time} seconds")

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            with chat_container:
                start_time = time.time()
                llm_response = llm_chain.run(st.session_state.user_question)
                end_time = time.time()
                st.session_state.user_question = ""
                print(f"Time taken for LLM for normal: {end_time - start_time} seconds")

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content) 
    
    save_chat_history()

if __name__ == "__main__":
    main()
