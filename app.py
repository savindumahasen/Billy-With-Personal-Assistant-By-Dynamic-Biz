import streamlit as st
from reportlab.lib.pagesizes import letter
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough

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

# Define the template for the chat prompt
template = """
Answer the question using the provided context only.

{question}

Context:
{context}

Answer:
"""

# Initialize the prompt with the template
prompt = ChatPromptTemplate.from_template(template)

# Function to set up the retriever from a PDF
def setup_retriever_from_pdf(pdf_filename):
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model)
    retriever = vector_store.as_retriever()
    return retriever

# Function to define the chain
def define_chain():
    chain = (
        RunnablePassthrough()  # Pass through input as is
        | llm  # Process with HuggingFaceHub
        | output_parser  # Parse the output
    )
    return chain

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
    retriever = setup_retriever_from_pdf(pdf_filename)
    st.success(f"Successfully loaded {pdf_filename} for processing.")
except Exception as e:
    st.error(f"Failed to load {pdf_filename}: {e}")
    st.stop()

# Streamlit input for user question
st.title("Dynamic Biz Smart Assistant 🤖")
question = st.text_input("Ask a question based on the content:")

if st.button("Generate Response"):  # Button to generate response
    if question:
        with st.spinner("🤔 Generating response..."):
            try:
                # Retrieve relevant context
                context_docs = retriever.get_relevant_documents(question)
                context_text = "\n".join([doc.page_content for doc in context_docs])  # Combine document content

                # Format the prompt with question and context
                formatted_prompt = prompt.format(question=question, context=context_text)

                # Define and invoke the chain
                chain = define_chain()
                response = chain.invoke(formatted_prompt)
                st.success("Response generated successfully!")
                st.write(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question!")
