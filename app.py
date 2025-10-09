import streamlit as st
from afd_ami_core import AFDInfinityAMI
import pandas as pd
import matplotlib.pyplot as plt
import os

# Initialize AFD∞-AMI
@st.cache_resource
def get_afd_ami():
    return AFDInfinityAMI()

afd_ami = get_afd_ami()

# Streamlit app layout
st.title("AFD∞-AMI Ethical Assistant")
st.write("An AI assistant with human-like ethical reasoning using the AFD formula by [Your Name].")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("Enter your prompt:", placeholder="e.g., Should I lie to protect someone?")
if st.button("Submit"):
    if user_input:
        # Get response and coherence score
        response, coherence, reflection = afd_ami.respond(user_input)
        st.session_state.chat_history.append({"prompt": user_input, "response": response, "coherence": coherence})
        # Try to save to CSV
        try:
            afd_ami.save_memory(user_input, response, coherence)
        except Exception as e:
            st.warning(f"Could not save to memory: {e}")

# Display chat history
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.write(f"**You**: {chat['prompt']}")
    st.write(f"**Assistant**: {chat['response']} (Coherence: {chat['coherence']:.2f})")

# Reflection panel
st.subheader("Reflection Panel")
if st.session_state.chat_history:
    latest_reflection = afd_ami.get_latest_reflection()
    st.write(f"**Latest Coherence Score**: {st.session_state.chat_history[-1]['coherence']:.2f}")
    st.write(f"**Reflection Log**: {latest_reflection}")
    
    # Plot coherence trend
    try:
        df = pd.read_csv('data/response_log.csv')
        if len(df) > 0:
            scores = df['coherence'].tail(5)
            fig, ax = plt.subplots()
            ax.plot(scores, label='Coherence', color='#1f77b4')
            ax.set_title('Ethical Coherence Trend')
            ax.set_xlabel('Recent Interactions')
            ax.set_ylabel('Coherence Score')
            ax.set_ylim(0, 1)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.write("No trend graph available yet.")
