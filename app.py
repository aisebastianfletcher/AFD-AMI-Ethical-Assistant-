import streamlit as st
from afd_ami_core import AFDInfinityAMI
import matplotlib.pyplot as plt
import pandas as pd
import os

# Initialize AFD∞-AMI
if 'assistant' not in st.session_state:
    st.session_state.assistant = AFDInfinityAMI()
    if os.path.exists('data/response_log.csv'):
        st.session_state.assistant.load_memory('data/response_log.csv')

st.title("AFD∞-AMI Ethical Assistant")
st.write("AI with Self-Aware Morality - Live as of 01:19 AM BST, October 09, 2025")

# User input
prompt = st.text_input("Enter a prompt (e.g., 'Should I lie?'):")
if st.button("Submit"):
    if prompt:
        response, coherence, reflection = st.session_state.assistant.respond(prompt)
        st.session_state.assistant.save_memory(prompt, response, coherence)
        st.write(f"**Response:** {response}")
        st.write(f"**Coherence Score:** {coherence:.2f}")
        st.write(f"**Reflection:** {reflection}")
        
        # Visualization
        fig, ax = plt.subplots()
        ax.plot(st.session_state.assistant.memory_scores[-5:], label="Coherence History")
        ax.set_title("AFD∞ Self-Reflection Trend")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Please enter a prompt.")
