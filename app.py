import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
import fitz  # PyMuPDF for alternative PDF loading
import speech_recognition as sr
from gtts import gTTS
import io

# Initialize the HuggingFace model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 500,
        "repetition_penalty": 1.2,
        "stop_sequence": ["\n"]
    },
    huggingfacehub_api_token="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"
)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize the output parser
output_parser = StrOutputParser()

# Define the template for the chat prompt with stricter restrictions
template = """
You must strictly answer the question based only on the context provided. Do not generate any information outside the provided content. 

The question must be answered based entirely on the context in the document. If the context does not contain relevant information, do not answer.

{question}

Context:
{context}

Answer:
"""

# Initialize the prompt with the template
prompt = ChatPromptTemplate.from_template(template)

# Function to set up the retriever from a PDF
def setup_retriever_from_pdf(pdf_filename):
    if not os.path.isfile(pdf_filename):
        raise FileNotFoundError(f"The PDF file '{pdf_filename}' does not exist.")
    
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = vector_store.as_retriever()
    return retriever

# Alternative function to extract text using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_filename):
    try:
        doc = fitz.open(pdf_filename)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF with PyMuPDF: {e}")
        return None

# Function to define the chain
def define_chain():
    chain = (
        RunnablePassthrough()  # Pass through input as is
        | llm  # Process with HuggingFaceHub
        | output_parser  # Parse the output
    )
    return chain

# Function for speech input (recognize speech)
def get_speech_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening for your question...")
        try:
            audio = recognizer.listen(source)
            question = recognizer.recognize_google(audio)
            st.write(f"You asked: {question}")
            return question
        except sr.UnknownValueError:
            st.error("Sorry, I didn't catch that. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Function for text-to-speech (speak response)
def speak_response(response):
    tts = gTTS(response, lang='en')
    tts.save("response.mp3")
    
    # Playing the response using Streamlit's audio player
    audio_file = open("response.mp3", "rb").read()
    st.audio(audio_file, format="audio/mp3")

# Streamlit UI configuration
st.set_page_config(
    page_title="Dynamic Biz - Intelligent Chat Assistant",
    page_icon="✨",
    layout="wide",
)

# Predefined PDF file name
pdf_filename = "output.pdf"

# Check if the PDF exists and load retriever
try:
    if not os.path.isfile(pdf_filename):
        st.error(f"The PDF file '{pdf_filename}' was not found!")
        st.stop()

    retriever = setup_retriever_from_pdf(pdf_filename)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
except Exception as e:
    pdf_text = extract_text_from_pdf(pdf_filename)
    if pdf_text:
        retriever = None
    else:
        st.error(f"Failed to load PDF even with alternative method: {e}")
        st.stop()

# Streamlit input for user question or voice input
st.title("Dynamic Biz Smart Assistant 🤖")

question_option = st.radio("Choose Input Method", ["Type a Question", "Ask by Voice"])

if question_option == "Type a Question":
    question = st.text_area("Ask a question based on the content:")
else:
    question = get_speech_input()

if st.button("Generate Response"):  # Button to generate response
    if question:
        with st.spinner("🤔 Generating response..."):
            try:
                if retriever:
                    context_docs = retriever.get_relevant_documents(question)
                    context_text = "\n".join([doc.page_content for doc in context_docs])  # Combine document content
                    
                    if not context_text.strip():
                        st.error("No relevant content found in the document for your question. Please ask something related.")
                        st.stop()
                else:
                    context_text = pdf_text
                
                if not context_text.strip():
                    st.error("No relevant content found in the document. Please ask a question related to the content.")
                    st.stop()
                
                formatted_prompt = prompt.format(question=question, context=context_text)
                chain = define_chain()
                response = chain.invoke(formatted_prompt)
                
                answer_start = response.find("Answer:")
                if answer_start != -1:
                    answer = response[answer_start + len("Answer:"):].strip()
                else:
                    answer = response
                
                st.success("Response generated successfully!")
                st.write(f"**Answer:** {answer}")

                # Speak the response
                speak_response(answer)
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question!")
