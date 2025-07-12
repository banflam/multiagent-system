import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
from huggingface_hub import login
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
)


login()

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage and returns the page content as markdown.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The markdown version of the webpage content.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()

        # Convert the HTML content to markdown
        markdown_content = markdownify(response.text).strip()
        
        # Remove all the multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        
        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
print(visit_webpage("https://archive.is/pzKXN")[:500])

model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools = [WebSearchTool(), visit_webpage],
    model = model,
    max_steps = 10,
    name="web_search_agent",
    description="Runs web searches for you.",
)

manager_agent = CodeAgent(
    tools = [],
    model = model,
    managed_agents = [web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

answer = manager_agent.run("If LLM training continues to scale up at the current rate until 2030, then what would the electric power in GW be required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")