# Key skills in the GenAI area for now are:
- Prompt engineering
- Understanding the importance of the context and techniques working with it
- The usage of accelerators and integration of various pieces of code (both EPAM and outside)

# Task
Demonstrate your understanding of those on a basic level and to deliver a Recommendation system based on GenAI.

1. Choose any public API or any API you have access to. For inspiration, you can use https://github.com/public-apis/public-apis?tab=readme-ov-file, https://github.com/ozlerhakan/mongodb-json-files/blob/master/datasets/books.json or https://github.com/emilydumas/mcs260fall2021/blob/main/samplecode/data/episodes.json, and grab a couple of dozen items of it.
2. Build a prompt that will guide a user through the process of choosing an item (like choosing a movie, a city, or whatever) and help him with possible doubts.
3. [Advanced & Optional]* Integrate it with the EPAM accelerator (https://epam-rail.com/) as a DIAL chat application with a system prompt designed on a previous step. Should be run locally. Please note that this accelerator is well-documented and contains examples - so it is not expected to be a guided exercise.

*The advanced & Optional part is for you if you want to target a more experienced team to challenge yourself and go deeper during the course.

# What Good prompt should include:
- Recommendation logic should be based MORE on the knowledge within the prompt, and LESS on GPT General knowledge.
- The bot instructed with this prompt should talk professionally, and make an impression of a "salesperson in a shop" 
  and guide a user through the process of choice.
- For key types of messages, a bot should give structured and predictable responses (like comparing two movies or 
  showing one item). Think of it as it is prepared to be rendered in special widgets.
- [Advanced] For certain use cases should be even pre-defined "text scripts" within a prompt.

The https://chat.lab.epam.com can be used as a chat engine, but since you're not working with private data here feel free to use public ChatGPT or any other LLM of choice.

# Deliverables expected:
- Prompt
- [Advanced & Optional] short video of a conversation with your prompt via your DIAL App launched locally and the 
  code of it.