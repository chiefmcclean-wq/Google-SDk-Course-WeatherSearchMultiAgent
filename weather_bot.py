"""Interactive Multi-Agent Weather & Search Bot using Google ADK.

Coordinates a multi-agent system consisting of:
1. 'root_agent': The Root Workflow Agent managing queries and loop-backs.
2. 'weather_agent': Specialized in weather forecasts, conditions, and storm alerts.
3. 'search_agent': Specialized in Google Search (`google_search`) for real-time web facts.

Protected by Google Model Armor guardrails and conversation logging.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure utf-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from google.adk.runners import InMemoryRunner
from google.genai import types
from weather_agent.agent import root_agent, weather_agent, search_agent
from weather_agent.tools import get_current_weather_for_location
from weather_agent.model_armor import screen_prompt, BLOCKED_RESPONSE_TEXT
from weather_agent.callbacks import ConversationLogger

# Load .env file
load_dotenv()


def print_banner():
    """Displays the welcome banner."""
    print("=" * 68)
    print("      Google ADK Multi-Agent Workflow -- Root, Weather & Search")
    print("=" * 68)
    print("Active Agents:")
    print("  * root_agent    : Coordinating Workflow Agent (routes & loop-backs)")
    print("  * weather_agent : Real-Time Weather Specialist (NWS & Open-Meteo)")
    print("  * search_agent  : Google Search Specialist (google_search)")
    print("-" * 68)
    print("Security: Protected by Google Model Armor Guardrails")
    print("Logging : Real-time conversation logging to 'conversation.log'")
    print("Commands: Type 'exit' or 'quit' at any time to end the session.")
    print("=" * 68)


def run_direct_demo_mode():
    """Runs a direct interactive demo using the tools when no API key is set."""
    print("\n[Notice] No GEMINI_API_KEY detected.")
    print("Starting in Direct Tool Interactive Mode with Model Armor guardrails")
    print("and conversation logging enabled. (To enable full multi-agent Gemini")
    print("reasoning and Google Search grounding, add GEMINI_API_KEY to .env).\n")

    session_id = "direct_demo_session"

    while True:
        try:
            user_input = input("\nEnter location for weather (or 'search <query>', or 'exit'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using the Multi-Agent Bot! Goodbye.\n")
                break

            # 1. Screen user prompt with Model Armor template
            verdict = screen_prompt(user_input)
            if verdict.is_blocked:
                ConversationLogger.log_blocked(
                    agent_name="root_agent",
                    session_id=session_id,
                    user_prompt=user_input,
                    category=verdict.violation_category or "Policy Violation",
                    reason=verdict.reason or "Prohibited content",
                    blocked_response=verdict.blocked_message,
                )
                print(f"\n[root_agent]: {verdict.blocked_message}")
                continue

            # 2. Log approved user prompt
            ConversationLogger.log_user(
                agent_name="root_agent",
                session_id=session_id,
                message=user_input,
            )

            # Check if search request in direct mode
            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                print(f"\n[search_agent]: In live mode with GEMINI_API_KEY, queries like '{query}' are handled via Google Search.")
                print(f"[search_agent]: Please add GEMINI_API_KEY to your .env to enable live search grounding.")
                continue

            print(f"\n[weather_agent]: [Fetching real-time weather for '{user_input}'...]")
            data = get_current_weather_for_location(user_input)

            if data.get("status") != "success":
                err_msg = data.get("message", "Location not found.")
                print(f"[weather_agent -> root_agent Loop-Back]: Could not resolve location. Returning to root.")
                print(f"[root_agent]: {err_msg}")
                ConversationLogger.log_model(
                    agent_name="root_agent",
                    session_id=session_id,
                    response=f"[Error] {err_msg}",
                )
                continue

            loc_name = data.get("location", user_input)
            forecast = data.get("forecast", {})
            alerts = data.get("alerts", {})

            summary_lines = [
                f"Location:     {loc_name}",
                f"Temperature:  {forecast.get('temperature', 'N/A')}",
            ]
            if "short_forecast" in forecast:
                summary_lines.append(f"Condition:    {forecast.get('short_forecast')}")
            elif "condition" in forecast:
                summary_lines.append(f"Condition:    {forecast.get('condition')}")
            if "wind" in forecast:
                summary_lines.append(f"Wind:         {forecast.get('wind')}")
            if "detailed_forecast" in forecast:
                summary_lines.append(f"Forecast:     {forecast.get('detailed_forecast')}")

            alert_list = alerts.get("alerts", [])
            if alert_list:
                summary_lines.append(f"Active Alerts ({len(alert_list)}):")
                for alt in alert_list:
                    summary_lines.append(f"   * [{alt.get('severity', 'Alert')}] {alt.get('event')}: {alt.get('headline')}")
            else:
                summary_lines.append("No active severe weather alerts.")

            formatted_summary = "\n".join(summary_lines)

            print("\n" + "=" * 55)
            print(formatted_summary)
            print("=" * 55)

            # 3. Log model response
            ConversationLogger.log_model(
                agent_name="weather_agent",
                session_id=session_id,
                response=formatted_summary,
            )

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Goodbye!")
            break


async def run_adk_agent_bot():
    """Runs the full Google ADK Multi-Agent event loop."""
    print_banner()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        run_direct_demo_mode()
        return

    # Initialize the ADK InMemoryRunner with the root workflow agent
    runner = InMemoryRunner(agent=root_agent)
    user_id = "default_user"

    # Create a persistent session for multi-turn conversation
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
    )

    print("[root_agent]: Hello! I am the Root Workflow Agent.")
    print("[root_agent]: I coordinate between our specialized Weather Agent and Google Search Agent.")
    print("[root_agent]: What weather or information can we assist you with today?\n")

    current_speaking_agent = "root_agent"

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n[root_agent]: Stay safe and have a wonderful day! Goodbye!\n")
                break

            # Format the message for ADK Runner
            new_message = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )

            current_speaking_agent = None

            # Drive the ADK Event Loop
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=new_message,
            ):
                # 1. Observe Agent Transfer / Loop-Back actions
                if event.actions and event.actions.transfer_to_agent:
                    dest = event.actions.transfer_to_agent
                    author = event.author or "unknown_agent"
                    reason_text = f" | Reason: {event.actions.transfer_reason}" if event.actions.transfer_reason else ""
                    if dest == "root_agent":
                        print(f"\n[🔄 Workflow Loop-Back]: '{author}' -> Looping back to '{dest}'{reason_text}")
                    else:
                        print(f"\n[🔄 Workflow Delegation]: '{author}' -> Delegating to '{dest}'{reason_text}")

                # 2. Inspect for tool invocations
                func_calls = event.get_function_calls()
                if func_calls:
                    for fc in func_calls:
                        args_summary = ", ".join(f"{k}='{v}'" for k, v in fc.args.items()) if fc.args else ""
                        print(f"\n[Tool Execution ({event.author})]: {fc.name}({args_summary})...")

                # 3. Extract generated text from event parts
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            author = event.author or "Agent"
                            if current_speaking_agent != author:
                                print(f"\n[{author}]: ", end="", flush=True)
                                current_speaking_agent = author
                            print(part.text, end="", flush=True)

                # 4. Check for errors in the event
                if event.error_code:
                    print(f"\n[ADK Error: {event.error_code}] {event.error_message}")

            print("\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[root_agent]: Session interrupted. Take care!\n")
            break
        except Exception as e:
            print(f"\n[Unexpected error]: {e}\n")


def main():
    """Main entrypoint."""
    try:
        asyncio.run(run_adk_agent_bot())
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")


if __name__ == "__main__":
    main()
