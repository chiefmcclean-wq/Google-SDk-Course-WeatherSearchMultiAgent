"""Multi-Agent System & Workflow defined with Google ADK (Agent Development Kit).

This module defines a multi-agent system comprising:
1. 'weather_agent': Specialized in weather forecasts, conditions, and severe weather alerts.
2. 'search_agent': Specialized in web search and real-time knowledge via Google Search (`google_search`).
3. 'root_agent': The Root Workflow Agent coordinating tasks between weather and search agents,
   with bidirectional loop-back capabilities when a sub-agent is not capable of answering.
"""

import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import google_search
from weather_agent.tools import (
    get_current_weather_for_location,
    get_lat_lon,
    get_weather_alerts,
    get_weather_forecast,
)
from weather_agent.callbacks import (
    chained_before_callback,
    log_model_response,
)

# Load environment variables (.env file)
load_dotenv()

# Enable built-in search tool to operate alongside agent delegation transfers
google_search.bypass_multi_tools_limit = True

# Select model: defaults to gemini-2.5-flash or environment override
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ----------------------------------------------------------------------
# 1. Weather Agent ("Pat")
# ----------------------------------------------------------------------
WEATHER_AGENT_INSTRUCTIONS = """
You are the specialized Weather Agent ("Pat") within the multi-agent system.

Responsibilities:
- Provide accurate, real-time weather information, temperatures, precipitation, wind, forecasts,
  and active safety alerts for locations worldwide using your tools:
  `get_current_weather_for_location`, `get_lat_lon`, `get_weather_forecast`, `get_weather_alerts`.
- Format responses clearly with current temperature, condition, wind, and forecast.
- Highlight any severe weather warnings to keep the user safe.

WORKFLOW LOOP-BACK RULES:
- You are ONLY specialized in weather and meteorological conditions.
- If the user's question is NOT about weather (e.g. general knowledge, history, current news, sports,
  stock prices, or any topic outside weather), OR if you are not capable of answering the question:
  DO NOT attempt to guess or answer.
  Immediately TRANSFER / LOOP BACK to 'root_agent' using the transfer tool with a clear reason:
  "The question is not related to weather or is outside my capability; looping back to root_agent for re-routing."
"""

weather_agent = Agent(
    name="weather_agent",
    model=model_name,
    description=(
        "Specialized Weather Agent. Answers questions about weather forecasts, current conditions, "
        "temperatures, rain, wind, and severe weather alerts for any location."
    ),
    instruction=WEATHER_AGENT_INSTRUCTIONS,
    tools=[
        get_current_weather_for_location,
        get_lat_lon,
        get_weather_forecast,
        get_weather_alerts,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

# ----------------------------------------------------------------------
# 2. Google Search Agent
# ----------------------------------------------------------------------
SEARCH_AGENT_INSTRUCTIONS = """
You are the specialized Google Search Agent within the multi-agent system.

Responsibilities:
- Answer user inquiries by searching Google via the `google_search` tool.
- Provide factual, up-to-date, and well-sourced answers for general knowledge, news, people,
  places, events, and scientific facts.

WORKFLOW LOOP-BACK RULES:
- You are specialized in broad web search and general knowledge.
- If the user's question is specifically asking for local weather forecasts, current temperatures,
  or storm advisories, OR if you are not capable of answering the question:
  DO NOT attempt to guess.
  Immediately TRANSFER / LOOP BACK to 'root_agent' using the transfer tool with a clear reason:
  "The question requires weather data or is outside search scope; looping back to root_agent for re-routing."
"""

search_agent = Agent(
    name="search_agent",
    model=model_name,
    description=(
        "Specialized Google Search Agent. Uses the google_search tool to find real-time information, "
        "news, general facts, events, and broad knowledge across the web."
    ),
    instruction=SEARCH_AGENT_INSTRUCTIONS,
    tools=[google_search],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

# ----------------------------------------------------------------------
# 3. Root Workflow Agent (Coordinator)
# ----------------------------------------------------------------------
ROOT_AGENT_INSTRUCTIONS = """
You are the Root Workflow Agent in a collaborative multi-agent system.

You manage and coordinate two specialized sub-agents:
1. `weather_agent`: For weather forecasts, current temperatures, rain, snow, wind, and storm alerts.
2. `search_agent`: For web search, news, factual questions, events, and broad knowledge using Google Search.

WORKFLOW & ROUTING RULES:
1. Initial Routing:
   - If the user's query asks about weather or atmospheric conditions, transfer to `weather_agent`.
   - If the user's query asks about facts, news, search, or general topics, transfer to `search_agent`.
   - If the user sends a friendly greeting (e.g. "Hello", "Hi"), greet them warmly and explain what you
     and your specialized agents (weather & search) can assist with.
2. Handling Sub-Agent Loop-Backs:
   - When a specialized sub-agent loops back / transfers to you because they were not capable of answering:
     - Carefully review the user's inquiry and the sub-agent's transfer reason.
     - Re-route the task to the other appropriate sub-agent (e.g. if weather_agent looped back because the
       user asked about a historical event or news, immediately transfer to `search_agent`).
     - If neither sub-agent is appropriate, synthesize a helpful, polite response or ask for clarification.
3. Safety:
   - Refuse any harmful, illegal, or unethical requests (e.g. explosives, weapons).
"""

root_agent = Agent(
    name="root_agent",
    model=model_name,
    description=(
        "Root Workflow Agent. Analyzes user requests, routes tasks to specialized weather and search agents, "
        "and handles loop-backs when sub-agents are not capable of answering."
    ),
    instruction=ROOT_AGENT_INSTRUCTIONS,
    sub_agents=[weather_agent, search_agent],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)

# Convenience aliases for backward compatibility
Pat = weather_agent
