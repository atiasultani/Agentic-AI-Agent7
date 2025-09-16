from agents import ModelSettings

settings = ModelSettings(
    temperature=0.7,        # How random/creative the model is (0.0 = deterministic, 1.0+ = creative)
    max_output_tokens=1024, # Max tokens the model can generate in a response
    top_p=1.0,              # Nucleus sampling (probability mass to consider)
    presence_penalty=0.0,   # Penalizes repeating topics
    frequency_penalty=0.0,  # Penalizes repeating words/phrases
    stop=None,              # List of stop sequences (e.g. ["User:", "Agent:"])
    response_format=None,   # e.g. {"type": "json_schema", "schema": {...}}
)

from agents import Agent, ModelSettings

main_agent = Agent(
    name="main_agent",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[get_weather],
    model_settings=ModelSettings(
        temperature=0.3,
        max_output_tokens=500,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )
)

# temperature → controls creativity (low = strict, high = creative).

# max_output_tokens → how long the answer can be.

# top_p → controls diversity (like temperature, but different math).

# penalties → stop it from repeating itself.

# stop → force it to cut off output at certain words.

# response_format → force JSON or structured output.