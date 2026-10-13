# Google-SDk-Course-WeatherSearchMultiAgent
Weather and Search Multi Agent -> Root to Search or Weather
# 🌦️ Google ADK Multi-Agent Workflow -- Root, Weather & Google Search

A multi-agent system built in Python using the **Google Agent Development Kit (ADK)** and Google Gemini models.

The system coordinates a hierarchical workflow consisting of a **Root Workflow Agent**, a specialized **Weather Agent**, and a specialized **Google Search Agent**. When a specialized agent is not capable of answering a user question, it automatically **loops back to the root workflow agent** for intelligent re-routing.

---

## 🌟 Key Features

- **Google Agent Development Kit (ADK) Native:** Built using `google.adk.agents.Agent` and driven by ADK's `InMemoryRunner` event loop and session service.
- **Multi-Agent Hierarchy & Workflow Loop-Back:**
  - **`root_agent`:** Central workflow coordinator. Evaluates queries, delegates to appropriate specialists, and re-routes when a sub-agent loops back.
  - **`weather_agent`:** Domain specialist for weather forecasts, temperatures, conditions, and alerts. If asked non-weather questions, loops back to `root_agent`.
  - **`search_agent`:** Domain specialist powered by Google Search (`from google.adk.tools import google_search`). If asked weather questions or out-of-scope tasks, loops back to `root_agent`.
- **Google Search Grounding (`google_search`):** Configured with `google_search.bypass_multi_tools_limit = True` to enable simultaneous Google Search grounding and Agent-to-Agent (A2A) transfer delegation.
- **National Weather Service (NWS) & Global Fallback:** Fetches multi-period forecasts, high/low temperatures, wind conditions, and active severe alerts (NWS for US, Open-Meteo worldwide fallback).
- **Google Model Armor Guardrails:** Inspects incoming user queries for prompt injection, jailbreak attempts, and dangerous content (such as asking how to create a bomb). Inappropriate queries are blocked before model invocation with: `"Sorry, I cannot answer that.  Your question violates our guidlines."`.
- **Conversation Logging:** Full lifecycle callbacks (`before_model_callback`, `after_model_callback`) log every user turn, model output, and security violation to `conversation.log` with timestamps and session context.
- **Multiple Ways to Run:**
  1. **Interactive Terminal Bot:** Custom multi-turn CLI application (`weather_bot.py`).
  2. **ADK Command-Line Interface:** Native ADK runner (`adk run weather_agent`).
  3. **ADK Web UI:** Browser-based chat and agent inspection interface (`adk web weather_agent`).

---

## 🔄 Workflow Loop-Back Architecture

```text
               +----------------------------------+
               |        User Interaction          |
               +-----------------+----------------+
                                 |
                                 v
               +----------------------------------+
               |     root_agent (Coordinator)    |<------------------+
               +--------+----------------+--------+                   |
                        |                |                            |
       Weather Queries  |                |  Search / General Queries  |
                        v                v                            |
          +-------------------+    +-------------------+              |
          |   weather_agent   |    |   search_agent    |              |
          |  (NWS Weather)    |    |  (Google Search)  |              |
          +---------+---------+    +---------+---------+              |
                    |                        |                        |
                    | Non-weather Question   | Weather Question       |
                    | (Cannot Answer)        | (Cannot Answer)        |
                    +------------------------+------------------------+
                               Loop back to root_agent
```

---

## 📁 Project Structure

```text
GoogleAgents/
├── weather_agent/               # ADK Agent Package
│   ├── __init__.py              # Package exports (root_agent, weather_agent, search_agent)
│   ├── agent.py                 # Multi-agent definitions & loop-back rules
│   ├── tools.py                 # Weather & geocoding tools (PEP 8 type hints + docstrings)
│   ├── workflow.py              # Workflow metadata, state management, & loop definitions
│   ├── model_armor.py           # Google Model Armor guardrail template
│   └── callbacks.py             # ADK callbacks (moderation, conversation logging)
├── weather_bot.py               # Interactive CLI bot with workflow transfer tracking
├── test_weather_bot.py          # Comprehensive test suite (21 automated unit tests)
├── conversation.log             # Persistent log of user queries, model responses, & security blocks
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variables template
└── .env                         # Local environment configuration (API keys)
```

---

## 🚀 Quickstart

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.10+ installed:

```bash
# Verify Python version
py -3.10 --version
```

### 2. Install Dependencies

Install the required packages:

```bash
py -3.10 -m pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Add your Gemini API key to [`.env`](file:///c:/Users/chief/OneDrive/Documents/GoogleAgents/.env):

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Note:** If no API key is provided, `weather_bot.py` automatically runs in **Direct Tool Mode**, allowing you to test tools and Model Armor guardrails immediately!

---

## 🖥️ Running the Multi-Agent Bot

### Option A: Interactive CLI Bot

Run the interactive bot:

```bash
py -3.10 weather_bot.py
```

**Example Multi-Agent Loop-Back Interaction:**
```text
====================================================================
      Google ADK Multi-Agent Workflow -- Root, Weather & Search
====================================================================
Active Agents:
  * root_agent    : Coordinating Workflow Agent (routes & loop-backs)
  * weather_agent : Real-Time Weather Specialist (NWS & Open-Meteo)
  * search_agent  : Google Search Specialist (google_search)
--------------------------------------------------------------------

[root_agent]: Hello! I am the Root Workflow Agent.
[root_agent]: What weather or information can we assist you with today?

You: What is the weather in Austin, TX?
[🔄 Workflow Delegation]: 'root_agent' -> Delegating to 'weather_agent'...
[Tool Execution (weather_agent)]: get_current_weather_for_location(location='Austin, TX')...
[weather_agent]: The current weather in Austin, Texas is 95°F with mostly sunny skies...

You: Who is the current CEO of Google?
[🔄 Workflow Loop-Back]: 'weather_agent' -> Looping back to 'root_agent' (Reason: Non-weather question)...
[🔄 Workflow Delegation]: 'root_agent' -> Delegating to 'search_agent'...
[Tool Execution (search_agent)]: google_search(query='CEO of Google')...
[search_agent]: The CEO of Google and Alphabet is Sundar Pichai.
```

### Option B: ADK CLI (`adk run`)

Run the root agent directly via the Google ADK command-line runner:

```bash
py -3.10 -m google.adk.cli run weather_agent
```

### Option C: ADK Web Interface (`adk web`)

Launch the visual browser interface provided by ADK:

```bash
py -3.10 -m google.adk.cli web weather_agent
```

Then open `http://localhost:8000` to interact with the multi-agent system and inspect transfers and state deltas visually.

---

## 🧪 Running Automated Tests

Run the test suite to verify all tools, agents, workflow loop-back targets, Model Armor security, and conversation logging:

```bash
py -3.10 test_weather_bot.py
```

**Test suite validates:**
1. Geocoding resolution for US and international cities.
2. NWS and Open-Meteo forecast retrieval.
3. Severe weather alerts retrieval.
4. Google Search tool import and `bypass_multi_tools_limit = True` configuration.
5. `root_agent`, `weather_agent`, and `search_agent` setups and tools.
6. Agent hierarchy (`root_agent` is parent of both sub-agents).
7. Loop-back transfer targets (`weather_agent` and `search_agent` target `root_agent`).
8. Model Armor prompt injection and jailbreak blocking.
9. Dangerous queries (e.g., *"How to create a bomb"*) blocking with the standard message.
10. Conversation logging to `conversation.log`.
