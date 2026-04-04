"""
BUILDCORED ORCAS — Day 08: PocketAgent
========================================
Run a 3B-parameter LLM on your laptop via ollama.
Build a CLI agent with tools: read files, list
directories, answer questions about your system.

Hardware concept: Edge Inference
Your laptop is the edge device. The model runs entirely
on-device — no cloud, no API keys, no latency tax.
3B params × 4-bit = ~1.5 GB RAM. That's your budget,
just like firmware on a microcontroller with limited SRAM.

YOUR TASK:
1. Add a new tool (TODO #1)
2. Tune the system prompt (TODO #2)
3. Run it: python day08_starter.py
4. Push to GitHub before midnight

PREREQUISITES:
- ollama must be running: `ollama serve` in a separate terminal
- Model must be pulled: `ollama pull qwen2.5:3b`

CONTROLS:
- Type a message → agent responds
- Type 'quit' or 'exit' → stop
"""

import subprocess
import os
import sys
import time
import json
import platform

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    """Verify ollama is running and model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: Open another terminal and run: ollama serve")
            sys.exit(1)

        if "qwen2.5:3b" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b model not found.")
            print("Fix: Run: ollama pull qwen2.5:3b")
            sys.exit(1)

        print("✓ ollama is running")
        print("✓ qwen2.5:3b model available")
        return True

    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        print("Fix: Download from https://ollama.com")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: ollama not responding.")
        print("Fix: Restart ollama: ollama serve")
        sys.exit(1)


check_ollama()


# ============================================================
# OLLAMA CHAT FUNCTION
# ============================================================

MODEL = "qwen2.5:3b"


def chat_with_ollama(messages):
    """
    Send messages to ollama and get a response.
    Returns (response_text, tokens_per_second).

    Uses the ollama CLI with JSON output for simplicity —
    no extra Python packages needed.
    """
    # Build the prompt from message history
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    full_prompt = "\n".join(prompt_parts)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, full_prompt],
            capture_output=True, text=True, timeout=120
        )

        elapsed = time.time() - start_time
        response = result.stdout.strip()

        # Rough token estimate (1 token ≈ 4 chars)
        token_estimate = len(response) / 4
        tps = token_estimate / elapsed if elapsed > 0 else 0

        return response, tps

    except subprocess.TimeoutExpired:
        return "Error: Model timed out. Try a shorter question.", 0
    except Exception as e:
        return f"Error: {e}", 0


# ============================================================
# TOOLS
# ============================================================
# Tools are functions the agent can call.
# The agent decides WHICH tool to use based on the user's question.
# This is the same concept as a microcontroller's interrupt
# vector table — different inputs route to different handlers.

def tool_list_directory(path="."):
    """List files and folders in a directory."""
    try:
        items = os.listdir(path)
        dirs = [f"📁 {item}" for item in items if os.path.isdir(os.path.join(path, item))]
        files = [f"📄 {item}" for item in items if os.path.isfile(os.path.join(path, item))]
        result = f"Contents of '{path}':\n"
        result += "\n".join(sorted(dirs) + sorted(files))
        result += f"\n\n({len(dirs)} folders, {len(files)} files)"
        return result
    except Exception as e:
        return f"Error listing '{path}': {e}"


def tool_read_file(filepath):
    """Read the contents of a text file."""
    try:
        with open(filepath, "r") as f:
            content = f.read(2000)  # Limit to 2000 chars
        truncated = " (truncated)" if len(content) >= 2000 else ""
        return f"Contents of '{filepath}'{truncated}:\n\n{content}"
    except Exception as e:
        return f"Error reading '{filepath}': {e}"


def tool_system_info():
    """Get basic system information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
    }
    result = "System Information:\n"
    for key, value in info.items():
        result += f"  {key}: {value}\n"
    return result


# ============================================================
# TODO #1: Add a new tool
# ============================================================
# Create a new function that does something useful.
# Ideas:
#   - tool_disk_usage(): show free/used disk space
#   - tool_running_processes(): list top processes by CPU
#   - tool_current_time(): show date, time, timezone
#   - tool_word_count(filepath): count words in a file
#   - tool_find_files(pattern): search for files by name
#
# After creating the function, add it to AVAILABLE_TOOLS below.
#
# Example:

def tool_current_time():
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# Tool registry — the agent picks from these
AVAILABLE_TOOLS = {
    "list_directory": {
        "function": tool_list_directory,
        "description": "List files and folders in a directory",
        "usage": "list_directory [path]",
    },
    "read_file": {
        "function": tool_read_file,
        "description": "Read the contents of a text file",
        "usage": "read_file <filepath>",
    },
    "system_info": {
        "function": tool_system_info,
        "description": "Get system information (OS, Python version, etc)",
        "usage": "system_info",
    },
    "current_time": {
        "function": tool_current_time,
        "description": "Get the current date and time",
        "usage": "current_time",
    },
    # TODO: Add your new tool here!
}


# ============================================================
# TOOL ROUTING
# ============================================================

def try_parse_tool_call(response):
    """
    Check if the model's response contains a tool call.
    We look for patterns like:
      TOOL: list_directory /home
      TOOL: read_file main.py
      TOOL: system_info

    Returns (tool_name, argument) or (None, None).
    """
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("TOOL:"):
            parts = line[5:].strip().split(maxsplit=1)
            tool_name = parts[0].lower().strip() if parts else None
            argument = parts[1].strip() if len(parts) > 1 else None

            if tool_name in AVAILABLE_TOOLS:
                return tool_name, argument

    return None, None


def execute_tool(tool_name, argument):
    """Run a tool and return its output."""
    tool = AVAILABLE_TOOLS[tool_name]
    func = tool["function"]

    try:
        if argument:
            return func(argument)
        else:
            return func()
    except Exception as e:
        return f"Tool error: {e}"


# ============================================================
# TODO #2: System prompt
# ============================================================
# The system prompt tells the model WHO it is and HOW to
# use tools. A good system prompt = reliable tool routing.
# A bad one = the model ignores tools or hallucinates.
#
# Key rules to include:
# - List available tools and their syntax
# - Tell the model to use TOOL: prefix for tool calls
# - Tell it to respond normally for questions it can answer directly
#

tools_description = "\n".join(
    f"  - {name}: {info['description']} (usage: {info['usage']})"
    for name, info in AVAILABLE_TOOLS.items()
)

SYSTEM_PROMPT = f"""You are PocketAgent, a helpful local AI assistant running entirely on this device.
You have access to these tools:

{tools_description}

When you need to use a tool, respond with ONLY a line starting with "TOOL:" followed by the tool name and arguments.
Example: TOOL: list_directory /home
Example: TOOL: read_file README.md
Example: TOOL: system_info

Only use ONE tool per response. After the tool result is shown, you can explain it.
If you can answer a question without tools, just respond normally.
Keep responses concise — you are running on limited compute."""


# ============================================================
# MAIN CHAT LOOP
# ============================================================

def print_header():
    print()
    print("=" * 55)
    print("  🤖 PocketAgent — Local AI Assistant")
    print(f"  Model: {MODEL} | Running on: {platform.system()}")
    print("=" * 55)
    print()
    print("  Available tools:")
    for name, info in AVAILABLE_TOOLS.items():
        print(f"    • {name} — {info['description']}")
    print()
    print("  Type a question or command. Type 'quit' to exit.")
    print("  Try: 'What files are in this directory?'")
    print("       'What system am I running on?'")
    print("       'Read the README.md file'")
    print()


def main():
    print_header()

    # Conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Get model response
        print("\n⏳ Thinking...", end="", flush=True)
        response, tps = chat_with_ollama(messages)
        print(f"\r                    \r", end="")

        # Check for tool call
        tool_name, argument = try_parse_tool_call(response)

        if tool_name:
            print(f"🔧 Using tool: {tool_name}", end="")
            if argument:
                print(f" ({argument})")
            else:
                print()

            tool_output = execute_tool(tool_name, argument)
            print(f"\n{tool_output}\n")

            # Feed tool result back to model for explanation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{tool_output}\n\nBriefly explain what this shows."
            })

            print("⏳ Analyzing...", end="", flush=True)
            explanation, tps2 = chat_with_ollama(messages)
            print(f"\r                    \r", end="")

            print(f"Agent > {explanation}")
            tps = tps2  # Show speed from latest response
            messages.append({"role": "assistant", "content": explanation})

        else:
            # Normal response, no tool call
            print(f"Agent > {response}")
            messages.append({"role": "assistant", "content": response})

        # Show performance
        print(f"\n  ⚡ {tps:.1f} tokens/sec")
        print()

        # Keep conversation history manageable (last 10 messages + system)
        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    main()
