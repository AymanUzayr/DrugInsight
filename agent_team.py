"""
Multi-Agent Orchestration: LeadDeveloper with Designer and Coder sub-agents.

This script implements the requested architecture using the `smolagents` library 
by Hugging Face, which perfectly supports "Managed Agents" (sub-agents) and routing.
It also provides tools for browser access (Designer) and code editor access (Coder).

To run this, first install the dependencies:
    pip install smolagents litellm duckduckgo-search

Set your API key before running (e.g. for OpenAI):
    $env:OPENAI_API_KEY="sk-..." (PowerShell)
      or
    export OPENAI_API_KEY="sk-..." (Linux/Mac)
"""

import os
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    ManagedAgent,
    LiteLLMModel,
    tool
)

# =======================================================================
# Custom Tools for the Coder Sub-Agent (Code Editor Access)
# =======================================================================
@tool
def read_file(file_path: str) -> str:
    """Reads the content of a local file.
    Args:
        file_path: The absolute or relative path to the file to read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Writes or overwrites content to a local file.
    Args:
        file_path: The path to the file to write to.
        content: The text content to write.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

@tool
def list_directory(directory_path: str = ".") -> str:
    """Lists files and directories in a given path.
    Args:
        directory_path: The path of the directory to list. Defaults to current directory.
    """
    try:
        return "\n".join(os.listdir(directory_path))
    except Exception as e:
        return f"Error listing directory {directory_path}: {e}"

def main():
    # =======================================================================
    # 1. Provide an LLM model back-end. 
    # Since you don't have a paid API key, you have two great FREE options:
    
    # OPTION A: Run locally using Ollama (100% Free, no keys needed)
    # 1. Download & install Ollama from https://ollama.com
    # 2. Run this in your terminal: `ollama run qwen2.5-coder:7b`
    # 3. Uncomment the line below:
    # model = LiteLLMModel(model_id="ollama/qwen2.5-coder:7b")

    # OPTION B: Use Hugging Face Serverless API (Free tier)
    # 1. Create a free account at https://huggingface.co
    # 2. Get a free access token from settings
    # 3. Set it as an environment variable: `setx HF_TOKEN "your_token"`
    # 4. Use the built-in HfApiModel below:
    from smolagents import HfApiModel
    model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")


    # =======================================================================
    # 2. Designer Sub-Agent (Browser & Research Access)
    # =======================================================================
    designer_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        name="designer",
        description="A Designer sub-agent that performs research using the web/browser to design solutions, fetch docs, and gather UI/UX or architectural patterns."
    )
    
    managed_designer = ManagedAgent(
        agent=designer_agent,
        name="designer",
        description="Call this agent exclusively for 'design' tasks, web research, fetching documentation, and planning software architectures."
    )

    # =======================================================================
    # 3. Coder Sub-Agent (Code Editor Access)
    # =======================================================================
    # Using CodeAgent also allows it to execute Python code logically.
    coder_agent = CodeAgent(
        tools=[read_file, write_file, list_directory],
        model=model,
        name="coder",
        description="A Coder sub-agent with code editor access that implements software, writes files, reads files, and executes Python code."
    )
    
    managed_coder = ManagedAgent(
        agent=coder_agent,
        name="coder",
        description="Call this agent exclusively for 'coding' tasks, implementation, reading or writing local files, and verifying code."
    )

    # =======================================================================
    # 4. Lead Developer Agent (Router / Orchestrator)
    # =======================================================================
    lead_developer = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[managed_designer, managed_coder],
        name="LeadDeveloper",
        description="You are the Lead Developer. Route 'design' tasks to the Designer and 'coding' tasks to the Coder. Coordinate the team to accomplish the user request."
    )

    # =======================================================================
    # Run the Team
    # =======================================================================
    print("Starting Multi-Agent Orchestration...")
    user_request = (
        "Design a simple, modern landing page with a hero section (research best practices), "
        "and then write the code to 'landing_page.html' in the current directory."
    )
    print(f"\nUser Request: {user_request}\n")
    
    # The LeadDeveloper will automatically figure out to call the designer for research,
    # read the design output, and then call the coder to create the file.
    final_response = lead_developer.run(user_request)
    
    print("\n--- Final Response ---")
    print(final_response)

if __name__ == "__main__":
    main()
