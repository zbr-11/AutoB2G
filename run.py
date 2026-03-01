import os
import subprocess
import sys
from pathlib import Path
import json
from agents.codebase_retrieval.agent import workflow_generation


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    CODEBASE_PATH = PROJECT_ROOT / "agents" / "codebase_retrieval" / "codebase.json"
    DATA_PATH = PROJECT_ROOT / "Grid_data"
    OUTPUT_PATH = PROJECT_ROOT / "output_grid" / "grid_sim_output"

    PROMPT = """ 
You are given a  building-grid co-simulation framework that couples CityLearn with a Pandapower-based grid network.
To reduce computational burden, it is assumed that the building power is replicated and injected into each network node.
Your task is to write the full simulation code so that it matches the user’s requirement.

Important constraints:
- The code template serves as a reference.
- Only choose necessary functions from the template.
- Do not add extra explanatory text or comments to the code.

"""

    USER_INSTRUCTION = """
Specific User Requirement:  
Test the centralized building control policies from both the building-side and grid-side perspectives. 
Use the SAC model trained for 2 episodes, with a quadratic reward that minimizes voltage deviations. 
Test the resulting strategy on an IEEE 33-bus distribution network with 24 buildings connected at each node. 
Add a reactive power shunt element at bus 13 that injects -1.6 MVAr. 
Compute the grid-side metrics, including bus voltage magnitudes, line loadings, 
and an N-1 contingency analysis using a voltage tolerance of 0.045 p.u. and a line loading threshold of 73\%. 
Output the results as both plots and CSV files.
"""

    ANSWER = workflow_generation(USER_INSTRUCTION,path=CODEBASE_PATH)
    TASK_DESCRIPTION = PROMPT + USER_INSTRUCTION + ANSWER

    # os.environ["OPENAI_API_KEY"] = "sk-xxxxxx_your_key_here"
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["DATA_PATH"] = str(DATA_PATH)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--task", TASK_DESCRIPTION,
        "--output", str(OUTPUT_PATH),
        "--mode", "lite",
    ]

    print(f"Output directory: {OUTPUT_PATH}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print("\n✅ SOCIA simulation completed successfully!")
        print(f"Results saved in: {OUTPUT_PATH}")
    else:
        print("\n❌ SOCIA simulation failed. Check logs in:")
        print(f"{OUTPUT_PATH / 'socia.log'}")

if __name__ == "__main__":
    main()
