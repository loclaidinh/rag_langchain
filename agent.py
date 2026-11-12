import os
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import RAG_WORKFLOW_INSTRUCTIONS, CHUNK_ANALYST_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS
from tools import search_documents, backend
from langchain.messages import HumanMessage

max_concurrent_analysts = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "="*80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts = max_concurrent_analysts
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "analyse one retreived documentation chunk file"
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

model = init_chat_model(
    model="deepseek/deepseek-v4.1-flash-1017",
    model_provider="openai",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

agent = create_deep_agent(
    model=model,
    tools=[search_documents],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent]
)

example_query = "what is the test case?"

result = agent.invoke(
    {"messages": [HumanMessage(content=example_query)]}
)

print(result["messages"][-1].content)