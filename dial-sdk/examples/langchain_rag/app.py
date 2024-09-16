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
            message = request.messages[-1]
            user_query = message.content or ""

            file_url = get_last_attachment_url(request.messages)
            file_abs_url = urljoin(f"{DIAL_URL}/v1/", file_url)

            if file_abs_url.endswith(".pdf"):
                loader = PyPDFLoader(file_abs_url)
            else:
                loader = WebBaseLoader(file_abs_url)

            # Create the download stage to show to the user the active process.
            # After the loading is complete, the stage will auto finished.
            with choice.create_stage("Downloading the document"):
                try:
                    documents = loader.load()
                except Exception:
                    msg = "Error while loading the document. Please check that the URL you provided is correct."
                    raise DIALException(
                        status_code=400, message=msg, display_message=msg
                    )

            # Show the user the total number of parts in the resource
            with choice.create_stage(
                    "Splitting the document into chunks"
            ) as stage:
                texts = text_splitter.split_documents(documents)
                stage.append_content(f"Total number of chunks: {len(texts)}")

            # Show the user start of calculating embeddings stage
            with choice.create_stage("Calculating embeddings"):

                openai_embedding = OpenAIEmbeddings(
                    model=EMBEDDINGS_MODEL,
                    openai_api_key=get_env("OPENAI_API_KEY"),
                )

                embeddings = CacheBackedEmbeddings.from_bytes_store(
                    openai_embedding,
                    embedding_store,
                    namespace=sanitize_namespace(openai_embedding.model),
                )

                docsearch = Chroma.from_documents(
                    texts, embeddings, collection_name=collection_name
                )

            # CustomCallbackHandler allows to pass tokens to the users as they are generated, so as not to wait for a complete response.
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


app = DIALApp(DIAL_URL, propagate_auth_headers=True)
app.add_chat_completion("echo", SimpleRAGApplication())


if __name__ == "__main__":
    # uvicorn.run('app:app', host='0.0.0.0', port=5001, reload=True)
    uvicorn.run(app, host='0.0.0.0', port=5002)
