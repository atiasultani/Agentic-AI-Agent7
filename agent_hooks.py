import os
import asyncio
from  openai import AsyncOpenAI
from agents import (Agent,
                    Runner,
                    RunConfig,
                    function_tool,
                    OpenAIChatCompletionsModel,
                    ModelSettings,
                    enable_verbose_stdout_logging,
                    RunContextWrapper,
                    set_tracing_disabled,
                    AgentHooks)

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME ="gemini-2.0-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=external_client,
)

config=RunConfig(
    model=model,
)
set_tracing_disabled(True)
enable_verbose_stdout_logging()


# Example of a Tool
@function_tool
async def get_weather(country: str) -> str:
    """Get the current weather for a given country."""
    # Simulate a weather API call
    return f"The current weather in {country} is sunny with a temperature of 25°C."

# Example of Agent Hooks

class HelloAgentHooks(AgentHooks):
    def __init__(self,lifeCycle_name:str):
        self.lifeCycle_name=lifeCycle_name

    async def on_start(self,context:RunContextWrapper,agent:Agent):
       
            print(f" \n\n lifecyle name :{self.lifeCycle_name} Agent {agent.name} is starting with context: {context}\n\n")
    async def on_end(self,context:RunContextWrapper,agent:Agent,output:str):
            print(f"Agent {agent.name} has finished with result: {output}")    
# hooks run in Runner.run   
# class Add(RunHooks):
#     async def on_agent_start(self, agent: Agent, input: str):
#         print(f"Agent {agent.name} is starting with input: {input}")

#     async def on_agent_end(self, agent: Agent, result: str):
#         print(f"Agent {agent.name} has finished with result: {result}")

news_agent = Agent( 
    name="news-agent",
    instructions="you are an agent that can provide news updates.",
    model=model,
    hooks=HelloAgentHooks("news_agent")
    )

main_agent = Agent(
    name="main_agent",
    instructions="you are a helpful assistant.",
    model=model,
    tools=[get_weather],
    hooks=HelloAgentHooks("main_agent"),
    handoffs=[news_agent],
    model_settings=ModelSettings(temperature=0.2)
)

async def main():
    print("Hello from agentic-ai-agent7!")

    result = await Runner.run(
            starting_agent=main_agent,
            input="What is the weather of city karachi,pakistan?",
            run_config=config)
    
    print("last_agent:",result.last_agent.name)
    print("Result:", result.final_output)
if __name__ == "__main__":
    asyncio.run(main())