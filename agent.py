import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
from huggingface_hub import login

login()

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"