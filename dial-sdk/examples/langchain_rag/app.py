"""
A simple RAG application based on LangChain.
"""
from urllib.parse import urljoin
from uuid import uuid4

import uvicorn
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_debug
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from aidial_sdk import DIALApp
from aidial_sdk import HTTPException as DIALException
from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response
from common.cfg import *
from utils import get_last_attachment_url, sanitize_namespace


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Please provide {name!r} environment variable")
    return value


DIAL_URL = get_env("DIAL_URL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
API_VERSION = os.getenv("API_VERSION", "2024-02-01")
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

set_debug(LANGCHAIN_DEBUG)

CONTENT_URL = "https://github.com/ozlerhakan/mongodb-json-files/blob/master/datasets/books.json"

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=0
)

embedding_store = LocalFileStore("./~cache/")


class CustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self, choice: Choice):
        self._choice = choice

    async def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
        self._choice.append_content(token)


class SimpleRAGApplication(ChatCompletion):
    async def chat_completion(
            self, request: Request, response: Response
    ) -> None:
        collection_name = str(uuid4())

        with response.create_single_choice() as choice:
            # Step 1: Track User Query and Message History
            user_query = request.messages[-1].content or ""
            message_history = [msg.content for msg in request.messages[:-1]]

            # Use message history to identify user preferences (e.g., genre, topic, etc.)
            user_preferences = self.extract_preferences_from_history(message_history)

            # Step 2: Load Document Content
            file_url = get_last_attachment_url(request.messages)
            file_abs_url = urljoin(f"{DIAL_URL}/v1/", file_url)

            if file_abs_url.endswith(".pdf"):
                loader = PyPDFLoader(file_abs_url)
            else:
                loader = WebBaseLoader(file_abs_url)

            with choice.create_stage("Downloading the document") as stage:
                try:
                    documents = loader.load()
                    stage.append_content("Document successfully downloaded.")
                except Exception:
                    msg = "Error while loading the document. Please check that the URL you provided is correct."
                    raise DIALException(
                        status_code=400, message=msg, display_message=msg
                    )

            # Step 3: Extract Document Content (titles, summaries, genres)
            with choice.create_stage("Analyzing document content") as stage:
                book_details = self.extract_book_details(documents)
                stage.append_content(f"Total number of books found: {len(book_details)}")

            # Step 4: Personalized Recommendations based on user history and document content
            recommended_books = self.recommend_books(user_query, user_preferences, book_details)

            # Structured response to show the recommendations
            with choice.create_stage("Providing recommendations") as stage:
                if recommended_books:
                    stage.append_content("Based on your preferences and the document content, here are some recommendations:")
                    for book in recommended_books:
                        stage.append_content(f"- {book['title']}: {book['summary']}")
                else:
                    stage.append_content("I couldn't find any specific matches based on your query, but here are some popular options:")
                    for book in book_details[:3]:  # Show a few random options if no exact match
                        stage.append_content(f"- {book['title']}: {book['summary']}")

            await response.aflush()

            # Continue with LLM-powered QA processing using document search
            texts = text_splitter.split_documents(documents)

            with choice.create_stage("Calculating embeddings") as stage:
                openai_embedding = OpenAIEmbeddings(
                    model=EMBEDDINGS_MODEL,
                    openai_api_key=get_env("OPENAI_API_KEY"),
                )
                embeddings = CacheBackedEmbeddings.from_bytes_store(
                    openai_embedding,
                    embedding_store,
                    namespace=sanitize_namespace(openai_embedding.model),
                )
                docsearch = Chroma.from_documents(texts, embeddings, collection_name=collection_name)
                stage.append_content("Embeddings successfully calculated.")

            llm = ChatOpenAI(
                model=CHAT_MODEL,
                openai_api_key=get_env("OPENAI_API_KEY"),
                temperature=0,
                streaming=True,
                callbacks=[CustomCallbackHandler(choice)],
            )

            await response.aflush()

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_kwargs={"k": 15}),
            )

            await qa.ainvoke({"query": user_query})

            docsearch.delete_collection()

    # Helper function to extract user preferences from history
    def extract_preferences_from_history(self, message_history):
        preferences = []
        for msg in message_history:
            if 'science fiction' in msg.lower():
                preferences.append('science fiction')
            elif 'fantasy' in msg.lower():
                preferences.append('fantasy')
            # Add more rules to detect other preferences (e.g., mystery, non-fiction, etc.)
        return preferences

    # Helper function to extract book details from loaded documents
    def extract_book_details(self, documents):
        book_details = []
        for doc in documents:
            # Access the text or other relevant attributes
            title = getattr(doc, 'title', 'Untitled')  # Replace 'title' with appropriate attribute or method
            summary = getattr(doc, 'summary', 'No summary available.')  # Replace 'summary' with appropriate attribute or method
            if not title or not summary:
                # If title or summary are not available, attempt to extract content
                text = getattr(doc, 'text', 'No content available.')
                title = 'Unknown Title'
                summary = text[:500]  # Example: Take the first 500 characters as a summary
            book_details.append({'title': title, 'summary': summary})
        return book_details

    # Helper function to recommend books based on query, user preferences, and document content
    def recommend_books(self, query, preferences, book_details):
        recommended = []
        for book in book_details:
            if any(pref in book['summary'].lower() for pref in preferences):
                recommended.append(book)
            elif query.lower() in book['title'].lower() or query.lower() in book['summary'].lower():
                recommended.append(book)
        return recommended



app = DIALApp(DIAL_URL, propagate_auth_headers=True)
app.add_chat_completion("echo", SimpleRAGApplication())


if __name__ == "__main__":
    # uvicorn.run('app:app', host='0.0.0.0', port=5001, reload=True)
    uvicorn.run(app, host='0.0.0.0', port=5002)
