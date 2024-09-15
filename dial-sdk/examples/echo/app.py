"""
A DIAL application which returns back the content and attachments
from the last user message.
"""
from common.cfg import *
import uvicorn
from aidial_sdk import DIALApp

from aidial_sdk.chat_completion import ChatCompletion, Request, Response

import openai


# openai_client = openai.AzureOpenAI(
#     azure_endpoint='https://ai-proxy.lab.epam.com',
#     azure_deployment='gpt-4',
#     api_key="dial_api_key",
#     api_version="2023-12-01-preview",
# )
openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

# ChatCompletion is an abstract class for applications and model adapters
class EchoApplication(ChatCompletion):
    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        # Get last message (the newest) from the history
        last_user_message = request.messages[-1]

        # Generate response with a single choice
        with response.create_single_choice() as choice:
            # Fill the content of the response with the last user's content
            chat_completion = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Whatever is given to you, please translate it to Polish",
                    },
                    {
                        "role": "user",
                        "content": last_user_message.content,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            completion = chat_completion.choices[0].message.content
            choice.append_content(completion or "")

            # choice.append_content(last_user_message.content or "")
            # if last_user_message.custom_content is not None:
            #     for attachment in (
            #         last_user_message.custom_content.attachments or []
            #     ):
            #         # Add the same attachment to the response
            #         choice.add_attachment(**attachment.dict())


# DIALApp extends FastAPI to provide a user-friendly interface for routing requests to your applications
app = DIALApp()
app.add_chat_completion("echo", EchoApplication())

# Run built app
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5001)
