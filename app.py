import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import requests
from ddgs import DDGS
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM (using Groq with Llama 3.1) - removed duplicate
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# Test 1: Groq LLM Connectivity (unchanged)
try:
    response = llm.invoke("What is the capital of France?")
    print("Groq LLM Test: Success")
    print("Response:", response.content.strip())
except Exception as e:
    print("Groq LLM Test: Failed")
    print("Error:", str(e))

# Test 2: OpenWeatherMap API Connectivity (unchanged)
try:
    city = "Bangalore"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") == 200:
        temp = response["main"]["temp"]
        condition = response["weather"][0]["description"]
        print("OpenWeatherMap Test: Success")
        print(f"Weather in {city}: {temp}°C, {condition}")
    else:
        print("OpenWeatherMap Test: Failed")
        print("Error:", response.get("message"))
except Exception as e:
    print("OpenWeatherMap Test: Failed")
    print("Error:", str(e))

# Test 3: DuckDuckGo Search Connectivity (unchanged)
try:
    with DDGS() as ddgs:
        results = [r for r in ddgs.text("latest AI chip news", max_results=3)]
    print("DuckDuckGo Search Test: Success")
    print("Top Result:", results[0]["title"] if results else "No results")
except Exception as e:
    print("DuckDuckGo Search Test: Failed")
    print("Error:", str(e))

# Updated prompt template: More explicit on format + summarization for news
prompt_template = PromptTemplate(
    input_variables=["input", "history"],
    template="""
You are a helpful assistant that can answer directly or use tools (Web Search, Weather).
IMPORTANT RULES:
- If the query is about CURRENT EVENTS, NEWS, or up-to-date info (e.g., "latest news"), ALWAYS use EXACTLY this format for the FIRST response: [Tool: Web Search: <exact query for search>]
- For weather, use EXACTLY: [Tool: Weather: <city>]
- After a tool result, summarize it into a detailed final answer (2-3 sentences, cite sources/dates).
- Otherwise, answer directly if confident.
- NEVER guess news—always tool first.

History: {history}
User Input: {input}
Respond with EITHER a tool call in EXACT format OR a detailed final answer.
"""
)

# Updated Web Search Tool: Add dates/sources if available, better formatting
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))  # List for reliability
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            date = r.get('date', 'No date')
            source = r.get('url', 'Unknown source')
            output.append(f"{i}. {r['title']} ({date}, {source})\n   {r['body'][:200]}...")
        return "\n".join(output)
    except Exception as e:
        return f"Web search error: {str(e)}"

# Weather Tool (unchanged)
def weather_check(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return f"Error: {response.get('message', 'Unable to fetch weather')}"
        temp = response["main"]["temp"]
        condition = response["weather"][0]["description"]
        return f"It's {temp}°C and {condition} in {city}."
    except Exception as e:
        return f"Weather check error: {str(e)}"

# Tool dictionary (unchanged)
tools = {
    "Web Search": web_search,
    "Weather": weather_check
}

# Updated ReAct Agent Loop: Flexible parsing with regex, better loop logic, cap history
def react_agent(input_query, max_iterations=3):
    history = ""
    for iteration in range(max_iterations):
        prompt = prompt_template.format(input=input_query, history=history)
        response = llm.invoke(prompt).content.strip()
        print(f"Iteration {iteration + 1} Raw LLM response: {repr(response)}")  # Debug

        # Improved tool call extraction with regex (handles variations)
        tool_match = re.search(r'\[Tool:\s*(Web Search|Weather)\s*:\s*(.*?)\s*\]', response, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_input = tool_match.group(2).strip()
            if tool_name in tools:
                result = tools[tool_name](tool_input)
                # Cap history to last tool only for brevity
                history = f"\nTool Used: {tool_name}\nInput: {tool_input}\nResult: {result}"
                print(f"Tool result: {result}")  # Debug
                # If this is iteration 2+, force final answer
                if iteration >= 1:
                    final_prompt = prompt_template.format(input="Summarize the tool result into a detailed answer.", history=history)
                    final_response = llm.invoke(final_prompt).content.strip()
                    return final_response
            else:
                return f"Error: Invalid tool {tool_name}"
        else:
            # No tool call: Assume direct answer (or final after tools)
            return response
    return "Error: Max iterations reached without a final answer."

# Gradio UI (unchanged)
def gradio_interface(query):
    return react_agent(query)

with gr.Blocks() as interface:
    gr.Markdown("# ReAct Agent")
    gr.Markdown("Ask about general knowledge, weather, or recent news!")
    
    input_text = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
    output_text = gr.Textbox(label="Response", lines=10)  
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=gradio_interface, inputs=input_text, outputs=output_text)

# Launch the UI
interface.launch()