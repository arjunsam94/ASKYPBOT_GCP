import os
import logging
# import argparse
import streamlit as st
from dotenv import load_dotenv
from google.cloud import firestore
from langchain.schema.document import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

# Load environent variables
load_dotenv()

# Environment variables
TEST_COLLECTION = os.environ.get("TEST_COLLECTION")
EMBED_MODEL = os.environ.get("EMBED_MODEL")
GEN_MODEL = os.environ.get("GEN_MODEL")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
DATABASE = os.environ.get("DATABASE")
COLLECTION = os.environ.get("COLLECTION")
TOP_K = os.environ.get("TOP_K")

# Configure logging
logging.basicConfig(level=logging.INFO)

# AI models
embedding = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
gen_model = ChatGoogleGenerativeAI(model=GEN_MODEL)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)

def query_rag(query_text: str):

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """


    client = firestore.Client(project=PROJECT_ID, database=DATABASE)

    query_vector = embedding.embed_query(query_text)

    vector_query = client.collection(TEST_COLLECTION).find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=int(TOP_K),
        distance_result_field="vector_distance",
    )
    
    docs = vector_query.stream()
    results = [result.to_dict() for result in docs if result.to_dict()['vector_distance'] < 1]

    context_text = "\n\n---\n\n".join([doc['content'] for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = gen_model.invoke(prompt)
    
    if not results:
        formatted_response = f"Response:\n\n{response_text.content}"
    else:
        seen_urls = set()
        lines = []
        for result in results:
            url = result['metadata'].get("source", None)
            if url not in seen_urls:
                seen_urls.add(url)
                lines.append(f"{result['metadata'].get("title",None)} : {url}")
        sources = "\n\n".join(lines)
        formatted_response = f"Response:\n\n{response_text.content}\n\nRelevant Docs:\n\n{sources}"    
    
    return formatted_response

def main():
    st.title("AskYP chatbot")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                st.markdown("**Assistant:**")
                st.write(query_rag(user_input))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()