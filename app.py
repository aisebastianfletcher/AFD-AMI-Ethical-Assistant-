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
st.write("AI with Self-Aware Morality - Live as of 01:35 AM BST, October 09, 2025")

# Layout: Chat on left, Reflection Panel on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Conversation")
    prompt = st.text_input("Enter a prompt (e.g., 'Should I lie?'):")
    if st.button("Submit"):
        if prompt:
            response, coherence, reflection, breakdown = st.session_state.assistant.respond(prompt)
            st.session_state.assistant.save_memory(prompt, response, coherence)
            st.write(f"**Prompt:** {prompt}")
            st.write(f"**Response:** {response}")
            st.write(f"**Coherence Score:** {coherence:.2f}")
            st.write(f"**Reflection:** {reflection}")

with col2:
    st.subheader("Reflection Panel")
    if st.session_state.assistant.memory_scores:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(st.session_state.assistant.memory_scores[-5:], label="Coherence")
        ax.set_title("Ethical Coherence Trend")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)
        with st.expander("Coherence Breakdown"):
            st.write(f"Harmony: {breakdown['harmony']:.2f}")
            st.write(f"Info Gradient: {breakdown['info_gradient']:.2f}")
            st.write(f"Oscillation: {breakdown['oscillation']:.2f}")
            st.write(f"Potential: {breakdown['potential']:.2f}")
