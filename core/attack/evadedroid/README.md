# EvadeDroid
This is a tool for generating Android adversarial examples in a black-box setting. EvadeDroid's pipeline uses a two-step approach (i) **preparation** (2) **manipulation**:

1. The preparation step is responsible for preparing an action set including a collection of gadgets (i.e., entry point, organ, and vein) extracted from benign Android apps that are publicly accessible. 

2. The manipulation is in charge of perturbing malware samples by applying a sequence of transformations gathered in the action set into malware samples over several iterations.

Note the pipeline's components of EvadeDroid have been recently moved from our research infrastructure to the repository; so, please let us know if you encounter some missing parts. Moreover, we are gradually completing the pipeline's guidelines.

## Installation
1. Install Python 3 (>= 3.6), Java 8 (Java SDK >= 1.8.0) and Android SDK on your machine. Your machine can be either Windows or Linux. Note for working with a Windows machine, you have to cut the following codes that are related to parallelizing processes in Python. These codes are used in different modules (e.g., main_pipeline.py). 

```python
import torch
mp = torch.multiprocessing.get_context('forkserver')

with mp.Pool(processes=config['nprocs_evasion']) as p:
                    p.starmap(..., zip(...)))

```

## Configuration
To configure the pipeline, go to setting.py and fill out the following path variables with right values:
- _absolute_project_path: The absolute root-path of the pipeline.
- _absolute_java_components_path: The absolute path of Java components used in Automate Software Transplantation and MaMaDroid.
- _absolute_graph_path: The absolute path for the graphs drawn based on the pipeline's outputs.

## Run the Pipeline
## Licensing
