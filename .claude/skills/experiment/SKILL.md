---
name: experiment
description: Setup an experimental folder for testing well-defined methods for achieving a specific goal. Only use when the user explicitly asks to begin an experiment.
---

# Experiment Skill

## Instructions

### Step 1: Ask for Experiment Goals and Metrics
- Ask for the name of the experiment from the user.
- Ask for a brief description of the experiment's goals
- Ask for the KPI or success metric they want to track for the experiment.

### Step 2: Define Methods to Test
- Ask how many methods they want to test in the experiment
For each method:
- ask for a name and a brief description of the method.

### Step 3: Setup the Experiment Folder
- Create a new folder `experiments/<experiment_name>` in the project root with the experiment name.
- Inside the folder, create a `CLAUDE.md` file with the experiment description and method details.
- Create a subfolder for each method with a placeholder file (i.e., `method.py`) for future implementation.

### Step 4: Implement Methods
For each method:
- Ask the user to provide related documentation, resources, code snippets, or a description to help implement the method.
- Save this information in the corresponding method folder in a `CLAUDE.md` file.
- Ask the user if you should implement a basic version of the method based on the provided information.
- If yes, implement a basic version of the method in the placeholder file.

### Step 5: Vary Implementations
For each method:
- ask the user if there are any minor implementation variations or parameterizations they want to vary
- If yes, add a `version` argument to the method implementation and ensure the main() function can run different versions based on this argument.

### Step 5: Setup Logging and Metric Tracking
- Ask the user how they want to log experiment results and track the KPI/metric.
- Implement a basic logging mechanism in each method's implementation to record runtimes, results, KPIs, and other metrics.
- Create a central results file (e.g., `results.csv`) in the experiment folder to aggregate results from all methods and variations.
- Create a runner script (e.g., `run_experiment.py`) that can execute all methods and variations, log results, and update the central results file.

### Step 6: Run Experiments
- Ask the user if they want to run the experiments now
- If now, execute the runner script, if not, save the experiment setup for later execution
- Ask the user if they want to go back to any previous steps to modify the experiment setup

### Step 7: Analyze Results
- Create a simple analysis script (e.g., `analyze_results.py`) that can read the central results file and generate basic visualizations or summaries of the experiment outcomes.
- Turn the output of the analysis script into a report format (e.g., `report.md`) that summarizes the findings and insights from the experiment.
- Ask the user if they want to go back and modify any methods or re-run the experiments based on the analysis results.
