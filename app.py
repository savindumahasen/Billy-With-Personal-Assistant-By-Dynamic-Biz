from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Allowed topics based on Dynamic Biz content
allowed_topics = [
    "company Offer", "Stories", "Work", "Blog", "Case Studies", "Contact Us",
    "Product Engineering", "Software Development", "Web Development", "Mobile Appliocations development", "IT Consulting",
    "Digital Growth", "SEO", "Digital Strategy", "Social Media Marketing", "Digital Advertising", "Email Marketing",
    "Creative", "Content Creation", "Branding", "UI/UX Designing"
]

# Restricting the model to only respond based on Dynamic Biz content
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant only capable of answering questions related to Dynamic Biz services, including 'What We Offer,' 'Our Story,' 'Our Work,' 'Blog,' 'Case Studies,' and 'Contact Us.' Do not provide information outside of this scope."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI for chatbot
st.set_page_config(
    page_title="Dynamic Biz - Intelligent Chat Assistant",
    page_icon="✨",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
        }
        .chat-header {
            text-align: center;
            font-size: 1.8em;
            font-weight: bold;
            color: #3b5998;
            margin-bottom: 20px;
        }
        .user-input {
            width: 100%;
            padding: 10px;
            border-radius: 25px;
            border: 1px solid #ddd;
            font-size: 1em;
        }
        .chat-bubble-user {
            text-align: right;
            margin: 10px 0;
        }
        .chat-bubble-user p {
            display: inline-block;
            background-color: #e9f5ff;
            padding: 10px 15px;
            border-radius: 15px;
            color: #333;
        }
        .chat-bubble-bot {
            text-align: left;
            margin: 10px 0;
        }
        .chat-bubble-bot p {
            display: inline-block;
            background-color: #f1f0f0;
            padding: 10px 15px;
            border-radius: 15px;
            color: #333;
        }
        .submit-button {
            margin-top: 10px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Main chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

st.markdown("<div class='chat-header'>Dynamic Bizz Smart Assistant 🤖</div>", unsafe_allow_html=True)

# Input box styled as a search bar
input_text = st.text_input("", placeholder="Search by keyword regarding the Dynamic Bizz", key="user_input")

# LLaMA Model setup (adjusting this for Hugging Face Hub)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",  # Replace with the actual LLaMA model ID from Hugging Face
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 500,
        "repetition_penalty": 1.2,
        "stop_sequence": ["\n"]
    },
    huggingfacehub_api_token="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"  # Ensure the API token is provided
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display chatbot interaction
if st.button("Submit", use_container_width=True):
    if input_text:
        # Check if the question relates to the allowed topics
        is_allowed = any(topic.lower() in input_text.lower() for topic in allowed_topics)
        
        if not is_allowed:
            response = "Error: I can only respond to questions related to the following topics: 'What We Offer', 'Our Story', 'Our Work', 'Blog', 'Case Studies', and 'Contact Us.' Please ask a question related to these topics."
        else:
            with st.spinner("🤔 Generating response..."):
                response = chain.invoke({'question': input_text})

        # User query bubble
        st.markdown(f"""
            <div class="chat-bubble-user">
                <p>{input_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # Bot response bubble
        st.markdown(f"""
            <div class="chat-bubble-bot">
                <p>{response}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")

st.markdown("</div>", unsafe_allow_html=True)  # Close chat container

# Display relevant business sections (optional)
st.markdown("""
    <div class='section'>
        <h2>What We Offer</h2>
        <p>We love to create, innovate, and inspire. Let us help you build the future!</p>
        <!-- Add further sections as needed -->
    </div>
""", unsafe_allow_html=True)
