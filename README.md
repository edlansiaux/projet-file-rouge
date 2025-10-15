# Red Thread Project – Simulation & Decision Support for Hospital Emergency Departments

This MSc/master project explores hospital emergency management, especially the imbalance between patient arrivals (demand) and available resources (supply).  
It aims to design decision-support systems (using AI, simulation, etc.) to reduce waiting times, improve resource allocation, and alleviate staff overload.

**Disclaimer**: This project is provided “as is”, for research and experimentation purposes.  
License: MIT (see [LICENSE](#license) file).

---

## Table of Contents

- [Context & Objectives](#context--objectives)  
- [Main Features](#main-features)  
- [Architecture & Components](#architecture--components)  
- [Installation & Usage](#installation--usage)  
- [Execution Examples](#execution-examples)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Context & Objectives

Hospital emergency departments face a recurrent issue: the mismatch between patient inflow and resource availability (staff, beds, equipment).  
This often leads to delays, dissatisfaction, and even risks for patients.  
The objectives of this project are to:

- Model patient flow and hospital resources through simulation.  
- Integrate decision-support modules (AI, heuristics, rules).  
- Test different scenarios (variable loads, admission policies, prioritization).  
- Evaluate improvements in waiting time reduction and efficiency.  

---

## Main Features

- Patient flow & queue simulation  
- Hospital resource modeling (staff, beds, equipment)  
- Decision-support module (AI / heuristics)  
- Output analysis (waiting times, resource occupancy, bottlenecks)  
- Scenario testing (load variations, strategies, prioritization rules)  

---

## Architecture & Components

The repository is organized as follows:

```yaml
project-file-rouge/
├── docs/                   # Documentation (concepts, methodology, diagrams)
├── scripts/                # Utility scripts, simulation launchers
├── fil_rouge_SMA_V1.ipynb  # Main experimentation / demonstration Jupyter notebook
├── LICENSE                 # MIT license
└── README.md               # (this file)
```
## Installation & Usage

Basic setup to clone, install, and run the project:

```sh
# 1. Clone the repository
git clone https://github.com/edlansiaux/projet-file-rouge.git
cd projet-file-rouge

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate     # Windows

# 3. Install dependencies (adjust if requirements.txt is available)
pip install -r requirements.txt

# 4. Run scripts or notebooks
# Example:
jupyter notebook fil_rouge_SMA_V1.ipynb
# Or via a script:
python scripts/run_simulation.py --config configs/scenario1.yaml
```

**Note**: Ensure that `requirements.txt` or `environment.yml` includes all dependencies (e.g. numpy, pandas, simpy, scikit-learn, matplotlib, etc.).

---

## Execution Examples

Some scenarios to try:

| Scenario                | Description                       | Parameters                           |
|-------------------------|-----------------------------------|--------------------------------------|
| Low load                | Few patients, baseline simulation | Low arrival rate, abundant resources |
| High load               | Heavy inflow of patients          | High arrival rate, limited resources |
| Prioritization strategy | Apply a triage/priority rule      | By severity, FIFO, etc.              |
| Strategy comparison     | Compare two allocation approaches | Heuristic A vs Heuristic B           |

Outputs may include: average waiting times, delay distributions, resource utilization rates, unserved patients, etc.

---

## Contributing 

Contributions are welcome! To contribute:

```yaml
steps:
  - Fork this repository
  - Create a feature branch: git checkout -b feature/my-idea
  - Make your changes and test them
  - Submit a pull request with a clear description
```

Please follow good practices: clean code, tests, documentation, and consistent architecture.

## License

This project is released under the MIT License. See the [LICENSE](#license) file for details.

## Authors & Acknowledgments

[Mohammed Berrajaa](https://github.com/medberrajaa) (main author)
[Alexandre Gauguet](https://github.com/GAUGUET) (main author)
[Hugo Kazzi]() (main author)
[Abdallah Lafendi]() (main author)
[Edouard Lansiaux](https://github/edlansiaux) (main author)
[Aurélien Loison](https://github.com/lsnaurelien) (main author)

Contributors via pull requests, reviews, or ideas

Special thanks to all contributors who help improve this project!


