import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit UI
st.title("Logical Fallacy Analyzer")

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key:",type="password")

# File upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# Initialize OpenAI
if api_key:
    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)

# Function to analyze logical fallacies
def analyze(chunk):
    messages = [
        SystemMessage(content="""You are an expert at spotting logical fallacies. Read the following text
                              looking for any logical fallacies. If you find a fallacy, name the fallacy and show the text 
                              snippet containing it."""),
        HumanMessage(content=chunk)
    ]
    response = chat(messages)
    return response.content

# Process the uploaded text file
if uploaded_file:
    with st.spinner('Processing...'):
        text = uploaded_file.read().decode('utf-8')

        # Split into chunks with overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        chunks = splitter.split_text(text)

        final_answer = ""
        chunk_num = 0

        for chunk in chunks:
            chunk_num = chunk_num + 1
            final_answer = final_answer + f"Section {chunk_num}: \n" + analyze(chunk) + "\n________________________________________________________\n\n"

    st.write(f"The text was broken into {len(chunks)} smaller sections and analyzed for logical fallacies.")
    st.write(final_answer)
