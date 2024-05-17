# Import all the necessary libraries
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

# ML_MODEL = "gpt-4o"
# ML_MODEL = "gpt-3.5-turbo"
ML_MODEL = "gpt-4-turbo"

load_dotenv()   # load environment variables from .env file

# Define which LLM to use
# gpt-4-11-06-preview is the GPT-4 turbo model launched by OpenAI at their Dev day in 2023
llm = ChatOpenAI(model=ML_MODEL, temperature=0)
# output = llm.invoke("What would be the AI equivalent of Hello World?")
# print(output)


# Add chat history to the Agent as short term memory
chat_history = []

# Add tools to the Agent to extent capabilities
tools = []

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful personal AI assistant named HAI. You have a geeky, clever, and accurate attitude."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Define the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
user_task = "What would be the AI equivalent of Hello World?"
output = agent_executor.invoke({"input": user_task, "chat_history": chat_history})
print(output['output'])