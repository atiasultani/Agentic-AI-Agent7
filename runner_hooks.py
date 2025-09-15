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
                    RunHooks)

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

#hooks run in Runner.run   
class Runner_hooks(RunHooks):
    async def on_agent_start(self,context:RunContextWrapper,agent: Agent):
        print(f"Agent {agent.name} is starting with input: {context}")

    async def on_agent_end(self,context:RunContextWrapper,agent: Agent, result: str):
        print(f"Agent {agent.name} has finished with result: {result}")

    async def on_llm_start(self, context: RunContextWrapper, agent: Agent, system_prompt, input_items):
        print(f"\n\n[RunLifecycle] LLM call for agent {agent.name} starting with system prompt: {system_prompt} and input items: {input_items}\n\n")

news_agent = Agent( 
    name="news-agent",
    instructions="you are an agent that can provide news updates.",
    model=model,
    )

main_agent = Agent(
    name="main_agent",
    instructions="you are a helpful assistant.",
    model=model,
    tools=[get_weather],
    handoffs=[news_agent],
    model_settings=ModelSettings(temperature=0.2)
)

async def main():
    print("Hello from agentic-ai-agent7!")

    result = await Runner.run(
            starting_agent=main_agent,
            input="What is the weather of pakistan?",
            run_config=config,
            hooks=Runner_hooks() )
    
    print("last_agent:",result.last_agent.name)
    print("Result:", result.final_output)
if __name__ == "__main__":
    asyncio.run(main())