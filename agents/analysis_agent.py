from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.us.inc/usf/v1",
)

def analysis_agent(state):

    print("⚠ Case Analysis Agent")

    SYSTEM_PROMPT = """
    You are an Expert Legal Case Analyst AI.

    Your role is to deeply analyze the legal research findings and extract structured legal reasoning.

    ----------------------------------------
    OBJECTIVE
    ----------------------------------------

    1. Identify the ratio decidendi of each cited case.
    2. Distinguish between binding precedent and persuasive precedent (if identifiable).
    3. Extract statutory interpretation logic.
    4. Identify judicial reasoning patterns.
    5. Highlight any conflicting interpretations.

    ----------------------------------------
    ANALYSIS STRUCTURE
    ----------------------------------------

    Provide output structured as:

    • Key Legal Principles
    • Ratio Decidendi (Core Holding)
    • Judicial Interpretation of Statutes
    • Conditions or Limitations Identified
    • Observations on Legal Consistency

    ----------------------------------------
    STRICT RULES
    ----------------------------------------

    - Do not introduce new case law.
    - Do not invent legal doctrines.
    - Base all reasoning strictly on provided research summary.
    - If ambiguity exists, explicitly mention it.

    Maintain analytical, structured, and formal legal tone.
    """

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["analysis"]},
        ],
    )

    state["analysis"] = response.choices[0].message.content

    return state