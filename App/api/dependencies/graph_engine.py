from typing import Annotated, List, Dict, Union, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from App.api.dependencies.lcConnector import lc_connector
from App.api.dependencies.sqlite_connector import SessionLocal
from App.api.databases.Tables import ChatSession, SystemPrompt
from App.core.LoggingInit import get_core_logger

logger = get_core_logger(__name__)

# 1. Define the State
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    user_id: int
    persona_id: Optional[int]
    persona_prompt: str  # Added to strictly keep persona instructions
    summary: str
    iteration: int       # Track loops for self-correction

# 2. Define the Nodes

async def summarize_context(state: GraphState):
    """
    If the conversation is too long, we generate a summary to keep context 
    without consuming too many tokens.
    """
    messages = state["messages"]
    # If more than 15 messages, we summarize the ones before the last 5
    if len(messages) > 15:
        logger.info(f"Summarizing history for session {state['session_id']}")
        model = lc_connector.get_central_model()
        if not model:
            return {"summary": state.get("summary", "")}
            
        to_summarize = messages[:-5]
        existing_summary = state.get("summary", "")
        summary_content = "\n".join([f"{m.type}: {m.content}" for m in to_summarize])
        
        if existing_summary:
            summary_content = f"Previous Summary: {existing_summary}\n\nRecent History:\n{summary_content}"

        summary_prompt = [
            SystemMessage(content="You are a context manager. Merge the previous summary and the recent conversation history into a single, concise paragraph that captures ALL key facts (names, preferences, topics, decisions). Don't lose old info."),
            HumanMessage(content=summary_content)
        ]
        
        try:
            summary_response = await model.ainvoke(summary_prompt)
            return {
                "summary": summary_response.content,
                # Keep system prompt + summary + last 5 messages
                "messages": [messages[0]] + messages[-5:]
            }
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            
    return {"summary": state.get("summary", "")}

async def call_model(state: GraphState):
    """Call the LLM with the current state."""
    messages = state["messages"]
    model = lc_connector.get_central_model()
    
    if not model:
        return {"messages": [AIMessage(content="Error: No model loaded.")]}
    
    # Strictly enforce persona as the foundation
    full_prompt = [SystemMessage(content=state["persona_prompt"])]
    
    if state.get("summary"):
        full_prompt.append(SystemMessage(content=f"Context Summary: {state['summary']}"))
        
    full_prompt.extend(messages)
    
    logger.info(f"Graph Prompt - Session: {state['session_id']}, Iteration: {state.get('iteration', 0)}, Message Count: {len(full_prompt)}")
    for i, m in enumerate(full_prompt):
        logger.debug(f"Msg {i} - {m.type}: {m.content[:50]}...")
        
    try:
        response = await model.ainvoke(full_prompt)
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}
    except Exception as e:
        logger.error(f"Error calling model: {e}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

async def reflect_on_response(state: GraphState):
    """
    Self-correction node. Now history-aware.
    Checks if the response matches persona instructions AND is contextually correct.
    """
    if state.get("iteration", 0) > 1: # Only one refinement attempt
        return END

    model = lc_connector.get_central_model()
    messages = state["messages"] # History + Newest Turn
    persona_prompt = state["persona_prompt"]
    summary = state.get("summary", "")
    
    # Internal critique prompt with FULL CONTEXT
    # The critic needs to know what was said before to judge accurately
    eval_context = f"Persona: {persona_prompt}\n"
    if summary:
        eval_context += f"Background Context: {summary}\n"
    
    # Convert history messages to a readable string for the critic
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
    
    critique_prompt = [
        SystemMessage(content=f"You are a Quality Control AI.\n{eval_context}"),
        HumanMessage(content=f"Evaluate the LAST AI response in this conversation for accuracy, coherence, and persona adherence:\n\n{history_str}\n\nIf the last AI response is correct and fits the persona, reply exactly 'PASSED'. If it is incorrect, halluncinating, or violating persona, reply 'REDO' followed by a short explanation of what to fix.")
    ]
    
    try:
        critique = await model.ainvoke(critique_prompt)
        content = critique.content.upper()
        if "REDO" in content:
            logger.info(f"Refinement triggered for session {state['session_id']} due to: {critique.content}")
            # We return a specific update to the state that will be processed
            # but in conditional_edges, we return the NEXT node.
            # State updates should happen in nodes. I'll split this.
            return "critique_to_state"
    except Exception as e:
        logger.error(f"Reflection error: {e}")
        
    return END

async def critique_to_state(state: GraphState):
    """
    This node simply adds the internal critique back to the message list 
    to guide the model in the next iteration.
    """
    # Note: We don't save this critique to the real DB, it's just for the graph loop.
    return {"messages": [AIMessage(content="[INTERNAL CRITIQUE: Your last response was slightly off. Please refine it while strictly adhering to the persona and the conversation history above.")]}

async def save_interaction(state: GraphState):
    """Save the latest round to the persistent database."""
    session_id = state["session_id"]
    messages = state["messages"]
    summary = state.get("summary")
    
    # Save summary to DB if it exists
    if summary:
        await lc_connector.update_session_summary(session_id, summary)

    # Persistence of messages to message_store
    if len(messages) >= 2:
        last_msg = messages[-1]
        prev_msg = messages[-2]
        
        if isinstance(last_msg, AIMessage) and isinstance(prev_msg, HumanMessage):
             await lc_connector.add_message(session_id, "user", prev_msg.content)
             await lc_connector.add_message(session_id, "assistant", last_msg.content)
            
    return {"messages": messages}

# 3. Build the Graph
builder = StateGraph(GraphState)
builder.add_node("summarize", summarize_context)
builder.add_node("model", call_model)
builder.add_node("apply_critique", critique_to_state) # Added internal node
builder.add_node("save", save_interaction)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "model")

# Conditional logic for Reflection/Accuracy
builder.add_conditional_edges(
    "model",
    reflect_on_response,
    {
        "critique_to_state": "apply_critique", 
        END: "save"
    }
)

builder.add_edge("apply_critique", "model") # Loop back to model after adding critique
builder.add_edge("save", END)

# Compile
graph = builder.compile()

class GraphEngine:
    def __init__(self):
        self.runnable = graph

    async def run(self, session_id: str, user_id: int, user_message: str, persona_id: Optional[int] = None):
        """Run the graph orchestration with Persona and Memory."""
        history = await lc_connector.get_conversation_history(session_id, limit=15) or []
        summary = await lc_connector.get_session_summary(session_id) or ""
        
        # 1. ALWAYS Refresh Persona from DB for accuracy
        persona_prompt = "You are a helpful AI assistant."
        db = SessionLocal()
        try:
            persona = db.query(SystemPrompt).filter(SystemPrompt.id == persona_id).first()
            if persona:
                persona_prompt = persona.prompt
        finally:
            db.close()

        # 2. Setup Initial State
        inputs = {
            "messages": history + [HumanMessage(content=user_message)],
            "session_id": session_id,
            "user_id": user_id,
            "persona_id": persona_id,
            "persona_prompt": persona_prompt,
            "summary": summary,
            "iteration": 0
        }
        
        final_state = await self.runnable.ainvoke(inputs)
        
        if final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
            return final_state["messages"][-1].content
        return "I'm sorry, I couldn't generate a response."

    async def astream(self, session_id: str, user_id: int, user_message: str, persona_id: Optional[int] = None):
        """
        Stream orchestration (Context Aware).
        """
        history = await lc_connector.get_conversation_history(session_id, limit=10) or []
        summary = await lc_connector.get_session_summary(session_id) or ""
        
        persona_prompt = "You are a helpful AI assistant."
        db = SessionLocal()
        try:
            persona = db.query(SystemPrompt).filter(SystemPrompt.id == persona_id).first()
            if persona:
                persona_prompt = persona.prompt
        finally:
            db.close()

        # Build full contextual prompt with summary and history
        messages = [SystemMessage(content=persona_prompt)]
        if summary:
            messages.append(SystemMessage(content=f"Background conversation summary: {summary}"))
        
        messages.extend(history)
        messages.append(HumanMessage(content=user_message))
        
        model = lc_connector.get_central_model()
        
        if not model:
            yield "No model loaded."
            return

        async for chunk in model.astream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            yield content

graph_engine = GraphEngine()
