source ../.venv/bin/activate
cd ../ && uvicorn dial-sdk.examples.langchain_rag.app:app --host 0.0.0.0 --port 5002
