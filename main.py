from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    InputGuardrailTripwireTriggered
)

import os
from dotenv import load_dotenv
from pydantic import BaseModel
import chainlit as cl

load_dotenv()
set_tracing_disabled(disabled=True)

# ✅ Gemini API Key (free key from makersuite)
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ✅ Async Client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ✅ Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # ✅ Free version
    openai_client=client
)

# ✅ INPUT GUARDRAIL

class PythonRelatedOutput(BaseModel):
    is_python_related: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="Input Guardrail",
    instructions="Check if the user's question is related to Python programming. Only return true if it is about Python.",
    output_type=PythonRelatedOutput,
    model=model
)

@input_guardrail
async def python_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python_related,
    )

# ✅ OUTPUT GUARDRAIL

class PythonOutput(BaseModel): 
    reasoning: str
    is_python: bool

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="Check if the output includes any Python related response.",
    output_type=PythonOutput,
    model=model
)

@output_guardrail
async def output_python_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, output)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_python,
    )


# ✅ MAIN AGENT

one_agent = Agent(
    name="PythonExpert",
    instructions="You are a Python expert. Only respond to Python programming questions.",
    model=model,
    input_guardrails=[python_guardrail],
    output_guardrails=[output_python_guardrail]
)

# ✅ Chainlit Start Message
@cl.on_chat_start
async def start():
    await cl.Message(content="👋 I’m a Python Expert Assistant. Ask me anything about Python programming!").send()

# ✅ Chainlit Message Handler
@cl.on_message
async def main(message: cl.Message):
    try:
        result = await Runner.run(one_agent, input=message.content)
        await cl.Message(content=result.final_output).send()

    except InputGuardrailTripwireTriggered:
        await cl.Message(content="⚠️ Sorry, I can only help with Python programming questions.").send()

    except OutputGuardrailTripwireTriggered:
        await cl.Message(content="⚠️ Output blocked: Your question was Python-related, but the response violated policy.").send()
