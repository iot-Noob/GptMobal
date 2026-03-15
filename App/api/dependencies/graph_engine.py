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
            
        history_to_summarize = messages[:-5]
        summary_prompt = [
            SystemMessage(content="Summarize the following conversation history briefly to preserve context:"),
            HumanMessage(content=str(history_to_summarize))
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
        
    try:
        response = await model.ainvoke(full_prompt)
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}
    except Exception as e:
        logger.error(f"Error calling model: {e}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

async def reflect_on_response(state: GraphState):
    """
    Self-correction node. Checks if the response matches persona instructions.
    Only runs once to avoid infinite loops.
    """
    if state.get("iteration", 0) > 1: # Already refined once
        return END

    model = lc_connector.get_central_model()
    last_response = state["messages"][-1].content
    persona_prompt = state["persona_prompt"]
    
    # Internal critique prompt
    critique_prompt = [
        SystemMessage(content=f"You are a quality controller. The user persona is: {persona_prompt}"),
        HumanMessage(content=f"Evaluate this response for accuracy and persona adherence: '{last_response}'. If it is good, reply 'PASSED'. If it needs improvement (e.g. out of character, incorrect), reply 'REDO' followed by the reason.")
    ]
    
    try:
        critique = await model.ainvoke(critique_prompt)
        if "REDO" in critique.content.upper():
            logger.info(f"Refinement triggered for session {state['session_id']}")
            # Add critique to history so model knows what to fix
            return "model" # Map to 'model' node in edges
    except:
        pass
        
    return END

async def save_interaction(state: GraphState):
    """Save the latest round to the persistent database."""
    session_id = state["session_id"]
    messages = state["messages"]
    
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
builder.add_node("save", save_interaction)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "model")

# Add conditional logic for Reflection/Accuracy
builder.add_conditional_edges(
    "model",
    reflect_on_response,
    {
        "model": "model", # Loop back to model for refinement
        END: "save"       # Or proceed to save
    }
)

builder.add_edge("save", END)

# Compile
graph = builder.compile()

class GraphEngine:
    def __init__(self):
        self.runnable = graph

    async def run(self, session_id: str, user_id: int, user_message: str, persona_id: Optional[int] = None):
        """Run the graph orchestration with Persona and Memory."""
        history = await lc_connector.get_conversation_history(session_id, limit=15) or []
        
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
            "summary": "",
            "iteration": 0
        }
        
        final_state = await self.runnable.ainvoke(inputs)
        
        if final_state["messages"] and isinstance(final_state["messages"][-1], AIMessage):
            return final_state["messages"][-1].content
        return "I'm sorry, I couldn't generate a response."

    async def astream(self, session_id: str, user_id: int, user_message: str, persona_id: Optional[int] = None):
        """
        Stream orchestration. 
        Note: The reflection loop is bypassed in raw streaming for lower latency.
        """
        history = await lc_connector.get_conversation_history(session_id, limit=10) or []
        
        persona_prompt = "You are a helpful AI assistant."
        db = SessionLocal()
        try:
            persona = db.query(SystemPrompt).filter(SystemPrompt.id == persona_id).first()
            if persona:
                persona_prompt = persona.prompt
        finally:
            db.close()

        # Build full contextual prompt
        messages = [SystemMessage(content=persona_prompt)] + history + [HumanMessage(content=user_message)]
        model = lc_connector.get_central_model()
        
        if not model:
            yield "No model loaded."
            return

        async for chunk in model.astream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            yield content

graph_engine = GraphEngine()
