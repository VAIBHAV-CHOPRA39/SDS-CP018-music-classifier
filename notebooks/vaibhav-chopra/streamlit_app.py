import streamlit as st

# Title of the app
st.title("Hello, World!")

# Display a simple message
st.write("Welcome to Streamlit!")

# Display some interactive widgets
name = st.text_input("What is your name?")
if name:
    st.write(f"Hello, {name}!")
