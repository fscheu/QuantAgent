# Message State Management in LangGraph

## Overview

This document explains how messages are managed in the QuantAgent multi-agent system, the architectural decisions behind the current implementation, and best practices for working with LangChain/LangGraph message states in parallel agent architectures.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [The Problem with Parallel Agents](#the-problem-with-parallel-agents)
- [Solution: Isolated Agent Contexts](#solution-isolated-agent-contexts)
- [Message Flow Diagram](#message-flow-diagram)
- [State Schema](#state-schema)
- [Agent-Specific Behavior](#agent-specific-behavior)
- [Follow-up Conversations](#follow-up-conversations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

QuantAgent uses a **fan-out/fan-in** graph structure where three analysis agents run in parallel, then converge to a decision agent:

```
START → [Indicator Agent, Pattern Agent, Trend Agent] → Decision Agent → END
```

Each agent serves a specific purpose:
- **Indicator Agent**: Computes and analyzes technical indicators (RSI, MACD, etc.)
- **Pattern Agent**: Identifies candlestick patterns using vision-capable LLMs
- **Trend Agent**: Analyzes trendlines and support/resistance levels
- **Decision Agent**: Synthesizes all reports into a trading decision (LONG/SHORT/HOLD)

## The Problem with Parallel Agents

### Initial Implementation (Incorrect)

Initially, all agents added their messages to the shared state:

```python
# ❌ INCORRECT: Each agent was doing this
def agent_node(state):
    messages = [
        SystemMessage("You are a specialized analyst..."),
        HumanMessage("Analyze the data...")
    ]
    response = llm.invoke(messages)

    # Adding messages to shared state
    return {
        "messages": messages,  # ← Problem: 3 agents × 2 messages each
        "report": report
    }
```

**Result**: The shared state accumulated multiple SystemMessages:

```python
state["messages"] = [
    SystemMessage("You are an indicator analyst..."),  # Indicator Agent
    HumanMessage("Analyze indicators..."),
    SystemMessage("You are a pattern analyst..."),     # Pattern Agent
    HumanMessage("Analyze patterns..."),
    SystemMessage("You are a trend analyst..."),       # Trend Agent
    HumanMessage("Analyze trends..."),
    SystemMessage("You are a trading decision maker..."), # Decision Agent
    HumanMessage("Make decision...")
]
```

### Why This is Wrong

1. **LLM Expectations**: Chat models expect a **single SystemMessage at the beginning**, not multiple intercalated ones
2. **Role Confusion**: Multiple SystemMessages confuse the LLM about its role and behavior
3. **Error**: LangGraph raises `INVALID_CONCURRENT_GRAPH_UPDATE` when parallel nodes try to update the same state key differently
4. **Inefficient**: Accumulates unnecessary messages that don't contribute to inter-agent communication

### LangChain Documentation Guidance

From LangChain's multi-agent documentation:

> "At the heart of multi-agent design is **context engineering** - deciding what information each agent sees."

SystemMessages should define the agent's role **for that specific agent's context**, not be shared across unrelated agents.

## Solution: Isolated Agent Contexts

### Correct Implementation

**Analysis Agents (Indicator, Pattern, Trend):**
- Use messages **locally** only for their LLM calls
- **Do NOT add messages to shared state**
- Communicate via **structured reports** (Pydantic models)

**Decision Agent:**
- Uses messages for its LLM call
- **DOES add messages to shared state**
- Enables follow-up conversations with context

### Code Pattern

#### Analysis Agents (Indicator, Pattern, Trend)

```python
def indicator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Create messages LOCALLY for this agent's LLM call
    agent_messages = [
        SystemMessage("You are a high-frequency trading analyst."),
        HumanMessage(f"Analyze these indicators: {state['kline_data']}")
    ]

    # Call LLM with local messages
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm_with_tools.with_structured_output(IndicatorReport)
    indicator_report = structured_llm.invoke(agent_messages)

    # Return ONLY the structured report, NOT messages
    return {
        "indicator_report": indicator_report,
        # NO "messages" key - messages stay local
    }
```

#### Decision Agent

```python
def trade_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Create messages for decision analysis
    agent_messages = [
        SystemMessage("You are a trading decision maker..."),
        HumanMessage(f"""
            Analyze these reports and make a decision:
            - Indicators: {state['indicator_report']}
            - Patterns: {state['pattern_report']}
            - Trends: {state['trend_report']}
        """)
    ]

    # Call LLM
    structured_llm = llm.with_structured_output(TradingDecision)
    trading_decision = structured_llm.invoke(agent_messages)

    # Return decision AND messages (for follow-up conversations)
    return {
        "final_trade_decision": trading_decision,
        "messages": agent_messages,  # ← Only Decision Agent adds messages
    }
```

## Message Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Graph Execution: Initial Analysis                          │
└─────────────────────────────────────────────────────────────┘

START
  ├─ kline_data: {...}
  ├─ time_frame: "4hour"
  └─ stock_name: "BTC"

    ┌──────────────────┬─────────────────┬──────────────────┐
    │                  │                 │                  │
    ▼                  ▼                 ▼                  │
┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ Indicator   │  │ Pattern      │  │ Trend        │       │ (Parallel)
│ Agent       │  │ Agent        │  │ Agent        │       │
└─────────────┘  └──────────────┘  └──────────────┘       │
                                                           │
LOCAL MESSAGES (not added to state):                      │
                                                           │
Indicator:                                                 │
  ├─ SystemMessage("You are HFT analyst...")              │
  └─ HumanMessage("Analyze indicators...")                │
                                                           │
Pattern:                                                   │
  ├─ SystemMessage("You are pattern recognition...")      │
  └─ HumanMessage("Analyze candlestick chart...")         │
                                                           │
Trend:                                                     │
  ├─ SystemMessage("You are trend analysis...")           │
  └─ HumanMessage("Analyze trendlines...")                │
                                                           │
    │                  │                 │                  │
    └──────────────────┴─────────────────┴──────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Decision Agent   │
                  └──────────────────┘

MESSAGES ADDED TO STATE:
  ├─ SystemMessage("You are a trading decision maker...")
  └─ HumanMessage("Based on these reports: ...")

                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Final State:                                             │
├──────────────────────────────────────────────────────────┤
│ messages: [                                              │
│   SystemMessage("You are a trading decision maker..."), │
│   HumanMessage("Based on these reports..."),            │
│   AIMessage("LONG with 0.85 confidence")                │ ← LLM response
│ ]                                                        │
│                                                          │
│ indicator_report: IndicatorReport(                      │
│   macd=1.23, rsi=65.4, confidence=0.78, ...             │
│ )                                                        │
│                                                          │
│ pattern_report: PatternReport(                          │
│   patterns_detected=["bullish_flag"], confidence=0.82   │
│ )                                                        │
│                                                          │
│ trend_report: TrendReport(                              │
│   trend_direction="upward", trend_strength=0.75         │
│ )                                                        │
│                                                          │
│ final_trade_decision: TradingDecision(                  │
│   decision="LONG", confidence=0.85, ...                 │
│ )                                                        │
└──────────────────────────────────────────────────────────┘
```

## State Schema

### IndicatorAgentState (agent_state.py)

```python
from typing import Annotated, List, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class IndicatorAgentState(TypedDict):
    """State type for the multi-agent trading system."""

    # Input data
    kline_data: Annotated[dict, "OHLCV data for analysis"]
    time_frame: Annotated[str, "Timeframe for the data (e.g., '4hour')"]
    stock_name: Annotated[str, "Symbol being analyzed (e.g., 'BTC')"]

    # Analysis agent outputs (structured reports)
    indicator_report: Annotated[IndicatorReport, "Technical indicator analysis"]
    pattern_report: Annotated[PatternReport, "Candlestick pattern analysis"]
    trend_report: Annotated[TrendReport, "Trend and support/resistance analysis"]

    # Decision agent output
    final_trade_decision: Annotated[TradingDecision, "Final trading decision"]

    # Messages (only used by Decision Agent for follow-up conversations)
    messages: Annotated[List[BaseMessage], add_messages, "Conversation history"]
```

### Key Points:

1. **`messages` field uses `add_messages` reducer**: Automatically merges new messages with existing ones
2. **Structured reports**: Pydantic models ensure type safety and validation
3. **Single responsibility**: Each field has a clear purpose

## Agent-Specific Behavior

### Indicator Agent (quantagent/indicator_agent.py)

**Purpose**: Compute technical indicators (RSI, MACD, ROC, Stochastic, Williams %R)

**Message Behavior**:
- Creates `SystemMessage` and `HumanMessage` **locally**
- Uses LangChain tool calling to compute indicators
- Returns structured `IndicatorReport`
- **Does NOT add messages to state**

**Implementation**:
```python
def indicator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent_messages = [
        SystemMessage(content="You are a high-frequency trading analyst..."),
        HumanMessage(content=f"Analyze indicators from: {kline_data}")
    ]

    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm_with_tools.with_structured_output(IndicatorReport)
    indicator_report = structured_llm.invoke(agent_messages)

    return {
        "indicator_report": indicator_report,
        # messages NOT included - stay local
    }
```

### Pattern Agent (quantagent/pattern_agent.py)

**Purpose**: Identify candlestick patterns using vision-capable LLMs

**Message Behavior**:
- Creates `SystemMessage` and `HumanMessage` with image content **locally**
- Uses vision LLM to analyze base64-encoded K-line charts
- Returns structured `PatternReport`
- **Does NOT add messages to state**

**Implementation**:
```python
def pattern_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent_messages = [
        SystemMessage(content="You are a pattern recognition assistant..."),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze this candlestick chart..."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
        ])
    ]

    final_response = graph_llm.invoke(agent_messages)
    pattern_report = PatternReport(...)

    return {
        "pattern_report": pattern_report,
        # messages NOT included - stay local
    }
```

### Trend Agent (quantagent/trend_agent.py)

**Purpose**: Analyze trendlines, support, and resistance levels

**Message Behavior**:
- Creates `SystemMessage` and `HumanMessage` with trend chart **locally**
- Uses vision LLM to analyze trendline-annotated charts
- Returns structured `TrendReport`
- **Does NOT add messages to state**

**Implementation**:
```python
def trend_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent_messages = [
        SystemMessage(content="You are a trend analysis assistant..."),
        HumanMessage(content=[
            {"type": "text", "text": "Analyze support/resistance..."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{trend_image}"}}
        ])
    ]

    final_response = graph_llm.invoke(agent_messages)
    trend_report = TrendReport(...)

    return {
        "trend_report": trend_report,
        # messages NOT included - stay local
    }
```

### Decision Agent (quantagent/decision_agent.py)

**Purpose**: Synthesize all reports into a final trading decision

**Message Behavior**:
- Creates `SystemMessage` and `HumanMessage` based on all reports
- Uses structured output to generate `TradingDecision`
- **DOES add messages to state** for follow-up conversations
- Only agent that contributes to shared message history

**Implementation**:
```python
def trade_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    indicator_report = state["indicator_report"]
    pattern_report = state["pattern_report"]
    trend_report = state["trend_report"]

    agent_messages = [
        SystemMessage(content="You are a trading decision maker..."),
        HumanMessage(content=f"""
            Based on these structured reports, make a decision:

            Indicators: {indicator_report.model_dump()}
            Patterns: {pattern_report.model_dump()}
            Trends: {trend_report.model_dump()}

            Issue LONG, SHORT, or HOLD.
        """)
    ]

    structured_llm = llm.with_structured_output(TradingDecision)
    trading_decision = structured_llm.invoke(agent_messages)

    return {
        "final_trade_decision": trading_decision,
        "messages": agent_messages,  # ← Added to state for follow-up
    }
```

## Follow-up Conversations

### Use Case

After the initial analysis, users may want to:
- Ask clarifying questions: "Why did you recommend LONG?"
- Explore alternatives: "What if RSI was 80 instead of 65?"
- Understand reasoning: "Which indicator was most important?"

### Implementation Pattern

```python
from quantagent.trading_graph import TradingGraph
from langchain_core.messages import HumanMessage

# 1. Initial Analysis
tg = TradingGraph(use_checkpointing=True)

initial_state = {
    "kline_data": df_dict,
    "time_frame": "4hour",
    "stock_name": "BTC",
}

config = {"configurable": {"thread_id": "user-123"}}
result = tg.graph.invoke(initial_state, config=config)

# 2. Display Results
print(f"Decision: {result['final_trade_decision'].decision}")
print(f"Confidence: {result['final_trade_decision'].confidence}")
print(f"Reasoning: {result['final_trade_decision'].reasoning}")

# 3. User Asks Follow-up Question
follow_up_state = {
    "messages": result["messages"] + [
        HumanMessage("Why did you recommend LONG instead of HOLD?")
    ],
    # Keep structured reports for context
    "indicator_report": result["indicator_report"],
    "pattern_report": result["pattern_report"],
    "trend_report": result["trend_report"],
}

# 4. Get Answer (only Decision Agent runs)
follow_up_result = tg.graph.invoke(follow_up_state, config=config)
print(follow_up_result["messages"][-1].content)
```

### With Checkpointing (Persistent Threads)

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Enable checkpointing
tg = TradingGraph(use_checkpointing=True)

# Initial analysis with thread
config = {"configurable": {"thread_id": "trading-session-456"}}
result = tg.graph.invoke(initial_state, config=config)

# Later: Resume conversation (state is persisted)
follow_up = tg.graph.invoke(
    {"messages": [HumanMessage("Explain the MACD signal")]},
    config=config  # Same thread_id
)
# LangGraph automatically loads previous messages from checkpoint
```

## Best Practices

### 1. Keep Agent Contexts Isolated

**✅ DO**: Use local messages for agent-specific LLM calls
```python
def my_agent(state):
    local_messages = [SystemMessage("You are..."), HumanMessage("...")]
    response = llm.invoke(local_messages)
    return {"report": response}
```

**❌ DON'T**: Add unrelated SystemMessages to shared state
```python
def my_agent(state):
    messages = [SystemMessage("You are..."), HumanMessage("...")]
    return {"messages": messages}  # ← Pollutes shared state
```

### 2. Use Structured Outputs for Inter-Agent Communication

**✅ DO**: Communicate via Pydantic models
```python
class IndicatorReport(BaseModel):
    macd: float
    rsi: float
    confidence: float
    reasoning: str

def indicator_agent(state):
    report = IndicatorReport(macd=1.23, rsi=65.4, ...)
    return {"indicator_report": report}
```

**❌ DON'T**: Pass analysis via messages
```python
def indicator_agent(state):
    analysis = "MACD is bullish, RSI is 65..."
    return {"messages": [AIMessage(analysis)]}  # ← Unstructured, hard to parse
```

### 3. Only Add Messages When Needed for Conversation

**✅ DO**: Add messages only for conversational endpoints
```python
def decision_agent(state):
    # This is the user-facing agent
    return {
        "decision": decision,
        "messages": agent_messages  # ← Enables follow-up questions
    }
```

**❌ DON'T**: Add messages from internal processing nodes
```python
def intermediate_processor(state):
    # Internal processing, no user interaction
    return {"messages": [AIMessage("Processing...")]}  # ← Unnecessary
```

### 4. Leverage `add_messages` Reducer

The `add_messages` reducer in LangGraph handles:
- **Appending** new messages to existing list
- **Updating** messages by ID (for edits)
- **Deduplication** based on message IDs

```python
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
```

### 5. Document Message Flow in Comments

```python
def my_agent(state):
    # Messages used LOCALLY for LLM call - not added to shared state
    # This agent communicates via structured MyReport output
    agent_messages = [SystemMessage("..."), HumanMessage("...")]

    result = llm.invoke(agent_messages)

    return {
        "my_report": result,
        # NO "messages" key - intentionally omitted
    }
```

## Troubleshooting

### Error: "Can receive only one value per step"

**Symptom**:
```
INVALID_CONCURRENT_GRAPH_UPDATE: At key 'messages': Can receive only one value per step.
```

**Cause**: Multiple parallel nodes are updating the `messages` key with different values.

**Solution**: Ensure only one node (or no nodes) in a parallel execution step updates `messages`. Analysis agents should NOT add messages.

**Fix**:
```python
# ❌ Before (causes error)
def parallel_agent(state):
    return {"messages": [...]}  # Multiple agents do this

# ✅ After (correct)
def parallel_agent(state):
    return {"report": ...}  # Only structured report, no messages
```

### Multiple SystemMessages in Conversation

**Symptom**: LLM responses are inconsistent or confused about its role.

**Cause**: Multiple SystemMessages in the message list from different agents.

**Solution**: Only the conversational endpoint (Decision Agent) should add SystemMessages to shared state.

### Messages Growing Too Large

**Symptom**: State size grows unbounded, causing performance issues.

**Solution**: Implement message trimming or summarization:

```python
from langchain.messages import RemoveMessage, trim_messages

def trim_old_messages(state):
    messages = state["messages"]

    # Keep only last 10 messages
    if len(messages) > 10:
        to_remove = messages[:-10]
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}

    return {}
```

### Lost Context in Follow-up

**Symptom**: Follow-up questions don't have context from initial analysis.

**Cause**: Not including structured reports when asking follow-up questions.

**Solution**: Always pass structured reports along with new messages:

```python
follow_up_state = {
    "messages": state["messages"] + [HumanMessage("New question")],
    # Include reports for context
    "indicator_report": state["indicator_report"],
    "pattern_report": state["pattern_report"],
    "trend_report": state["trend_report"],
}
```

## References

### LangChain/LangGraph Documentation
- [Messages in LangChain](https://docs.langchain.com/oss/python/langchain/messages)
- [LangGraph StateGraph](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [Multi-agent Systems](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [add_messages Reducer](https://docs.langchain.com/oss/python/langgraph/use-graph-api#process-state-updates-with-reducers)

### Project Files
- State definition: `quantagent/agent_state.py`
- Graph setup: `quantagent/graph_setup.py`
- Indicator agent: `quantagent/indicator_agent.py`
- Pattern agent: `quantagent/pattern_agent.py`
- Trend agent: `quantagent/trend_agent.py`
- Decision agent: `quantagent/decision_agent.py`

### Related Documentation
- [Configuration Management](./CONFIGURATION.md) - Environment-based configuration
- [Migrations Guide](./MIGRATIONS.md) - Database setup and migrations
- [Testing Patterns](./TESTING_PATTERNS.md) - Testing best practices

## Summary

**Key Takeaways**:

1. ✅ **Analysis agents use messages locally** - Not added to shared state
2. ✅ **Decision agent adds messages to state** - Enables follow-up conversations
3. ✅ **Agents communicate via structured reports** - Type-safe with Pydantic
4. ✅ **Single SystemMessage in shared state** - From Decision Agent only
5. ✅ **`add_messages` reducer handles merging** - Automatic message accumulation

This architecture ensures:
- **Clean message history** without role confusion
- **Efficient state management** with minimal overhead
- **Type-safe inter-agent communication** via structured models
- **Conversational capabilities** for user interaction
- **Scalability** for adding new analysis agents

For questions or issues with message state management, refer to the [Troubleshooting](#troubleshooting) section or consult the LangGraph documentation.
