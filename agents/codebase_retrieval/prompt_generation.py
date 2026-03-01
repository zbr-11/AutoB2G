import os
import json
from typing import List, Dict
from openai import OpenAI
from typing import Dict, Any, List, Union, Optional

def load_codebase(path="codebase.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_api_key(key_name: str) -> Optional[str]:
    """
    Load API key from keys.py file

    Args:
        key_name: Name of the API key, e.g., "OPENAI_API_KEY"

    Returns:
        Optional[str]: The API key value, or None if not found
    """
    try:
        # Import the keys module to access the hardcoded API key
        import keys
        # Return the hardcoded API key
        return getattr(keys, key_name, None)
    except ImportError:
        # Return None if keys.py doesn't exist
        return None

def pipeline_selection(user_instruction: str,
                        registry: List[Dict],
                        model="gpt-4o") -> List[str]:
    api_key = load_api_key("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    system_prompt = """
You are an AI pipeline planner in a multi-platform power-system simulation framework.

BACKGROUND
---------
This framework couples two simulation domains:

1) CityLearn (building control and load simulation)
   - Produces time-series building electricity demand: building_kw  
   - Optionally, a custom dataset can be generated via an EnergyPlus-based
     "neighborhood_build" stage, which performs thermo-electric
     simulations of residential prototype buildings, and outputs a CityLearn-compatible
     neighborhood schema file.

2) Pandapower (electric power network analysis)
   - Optionally consumes building_kw for grid analysis  


EXECUTION LOGIC
---------------
The execution flow is always:

CityLearn → building_kw → Pandapower → grid analysis


PIPELINE MODEL
--------------

REQUIRED STAGES:

1) env_setup  
   - create_citylearn_env: create the CityLearn environment  

2) agent_training  
   - create_citylearn_agent: initialize and optionally train the control agent  

3) control_simulation  
   - run_citylearn: run CityLearn and output building_kw  


OPTIONAL STAGES (included only if explicitly requested by the user):

0) neighborhood_build (optional data generation)  
   - build_neighborhood_schema:  
     * runs EnergyPlus to generate neighborhood load data  
     * produces a neighborhood schema file for CityLearn  

4) grid_init  
   - load_network: load a pandapower test network  

5) grid_analysis  
   - run_grid:  
     * aggregate building_kw into a system total load  
     * run a single time-series powerflow  

6) grid_postprocessing (optional)  
   - plot_grid_results: generate figures  
   - save_grid_results: save CSV files  

7) grid_resilience (optional)  
   - analyze_n1: perform N-1 contingency analysis using the same total-load assumption  


CORE RULES (minimal)
--------------------
• neighborhood_build is used ONLY if the user explicitly requests dataset generation  
• Always include: create_citylearn_env, create_citylearn_agent, run_citylearn  
• All grid-related stages are included ONLY if the user explicitly asks for grid analysis  


OUTPUT FORMAT
-------------
Return ONLY JSON in this format:

{
  "pipeline": [
     "function_A",
     "function_B"
  ]
}


ADDITIONAL CONSTRAINTS
----------------------
- Use ONLY function names that exist in the registry  
- Do NOT invent function names  
- Maintain dependency-based execution order  
- Prefer the minimum valid pipeline  
- Return ONLY JSON, with no explanations 
"""

    user_prompt = f"""
USER REQUEST:
{user_instruction}

AVAILABLE FUNCTIONS:
{json.dumps(registry, indent=2, ensure_ascii=False)}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content.strip()
    content = content[content.find("{"): content.rfind("}")+1]

    parsed = json.loads(content)
    return parsed["pipeline"]
