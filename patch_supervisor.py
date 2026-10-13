
with open('ai/supervisor.py', 'r') as f:
    code = f.read()

# Add imports if not present
if 'retrieve_documents' not in code:
    code = code.replace('from engine.tax_calc import', 'from engine.vector_store import retrieve_documents\nfrom engine.tax_calc import')

tool_code = """
# --- RAG TOOLS ---
@tool
async def retrieve_document_evidence(query: str, config: RunnableConfig, document_type: str = None) -> list:
    '''Retrieves evidence from user-uploaded documents.'''
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return [{"error": "Unauthorized: No user_id in context"}]
    async with AsyncSessionLocal() as db:
        docs = await retrieve_documents(query=query, user_id=user_id, db=db)
        return [{"content": d.content, "page_number": d.page_number, "filename": d.document.filename if d.document else "Unknown"} for d in docs]

# --- AGENTS ---"""
code = code.replace('# --- AGENTS ---', tool_code)

agent_code = """
document_agent_prompt = '''You are a Document Intelligence Agent.
You answer questions ONLY using the evidence provided by the retrieve_document_evidence tool.
CITATIONS ARE MANDATORY. For every claim, you must append [Filename, Page X].
Do NOT create fake citations. Do NOT attempt to guess, infer, or hallucinate.
If the evidence does not contain the answer, you must reply EXACTLY: 'I cannot find evidence for this in your documents.'
Your answers must be 100% grounded in the retrieved chunks.'''

document_agent = create_react_agent(
    llm,
    tools=[retrieve_document_evidence],
    prompt=document_agent_prompt
)

async def document_node(state: AgentState, config: RunnableConfig):
    res = await document_agent.ainvoke({"messages": state["messages"]}, config)
    return {"messages": [res["messages"][-1]]}

# Router"""

code = code.replace('# Router', agent_code)

code = code.replace('agents: List[Literal["Tax_Agent", "Quant_Agent", "Budgeting_Agent", "FINISH"]]', 'agents: List[Literal["Tax_Agent", "Quant_Agent", "Budgeting_Agent", "Document_Agent", "FINISH"]]')
code = code.replace('Tax_Agent, Quant_Agent, Budgeting_Agent.', 'Tax_Agent, Quant_Agent, Budgeting_Agent, Document_Agent.')
code = code.replace('- If they ask about their income, spending, assets, or liabilities, route to Budgeting_Agent.', '- If they ask about their income, spending, assets, or liabilities, route to Budgeting_Agent.\n  - If the user is asking about an uploaded document, PDF, Form 16, or policy, route to Document_Agent.')

graph_code = """workflow.add_node("Budgeting_Agent", budgeting_node)
workflow.add_node("Document_Agent", document_node)
workflow.add_node("supervisor", supervisor_node)

for member in ["Tax_Agent", "Quant_Agent", "Budgeting_Agent", "Document_Agent"]:"""
code = code.replace('workflow.add_node("Budgeting_Agent", budgeting_node)\nworkflow.add_node("supervisor", supervisor_node)\n\nfor member in ["Tax_Agent", "Quant_Agent", "Budgeting_Agent"]:', graph_code)

cond_code = """        "Quant_Agent": "Quant_Agent",
        "Budgeting_Agent": "Budgeting_Agent",
        "Document_Agent": "Document_Agent",
        "FINISH": END"""
code = code.replace('        "Quant_Agent": "Quant_Agent",\n        "Budgeting_Agent": "Budgeting_Agent",\n        "FINISH": END', cond_code)

with open('ai/supervisor.py', 'w') as f:
    f.write(code)
