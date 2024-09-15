source .venv/bin/activate
uvicorn dial-sdk.examples.echo.app:app --host 0.0.0.0 --port 5000
