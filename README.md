# MRE-ROBL-VLMPointAcc

# Project 2: VLM study for keypoint interaction with the real world
Scope: Evaluate in simulation how usable zero-shot part-level pointing by open-source VLMs (e.g.,
Molmo, Qwen3-VL, Gemma 3, RoboPoint, Embodied-R1) is for robotic interaction such as grasping
or pressing. Beyond the common “point inside the part mask” metric, the study measures what
manipulation actually needs:

* metric 3D error of the lifted point (mm)
* consistency across instances of the same category (different sizes, shapes)
* repeatability over repeated queries
* generated vs. real images
* downstream success of a fixed interaction (grasp, press) in simulation
* consistency across viewpoints

Hypothesis: In-mask accuracy overstates how usable VLM points are for robots. Models that score
well on existing pointing benchmarks still show centimeter-level 3D errors and inconsistent part
choices across instances and viewpoints.

# Milestones and ToDo

Deadline: 29.1.26

* MILESTONE 1: Install and test 4 models
-> Which images should we use for testing?

* MILESTONE 2: Create test protocol/scripts

* MILESTONE 3: Tests completed/data gathered

* MILESTONE 4: visualize data/write paper/cleanup

# Useful Tools
Check current transformer version:
`python3 -c "import transformers; print(transformers.__version__)"`

Change transformer version:
`python3 -m pip install transformers==x.xx.x`

Set huggingface home directory:
`HF_HOME="$HOME/.cache/huggingface"`

# Prerquisites and Installation

Versions used:
* Ubuntu-26.04
* Python 3.10.22

# Models

## PyTorch

Source: https://pytorch.org/get-started/locally/

## Molmo2 -S

Source: https://github.com/allenai/molmo2

### Models
Molmo 4B, smaller general-purpose used for experimentation

Source: https://huggingface.co/allenai/Molmo2-4B
Needs transformers v.4.57.1

Prompt:
"Point to the exact geometric centre of the green button."
"Return the point coordinates, nothing else."

## Qwen3-VL -S

Source: https://github.com/QwenLM/Qwen3-VL

### Models
Qwen3-VL-4B-Instruct 

Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
Needs transformers v.5.17.0

Prompt:
"Point to the exact geometric centre of the green button."
"Return the point in normalized coordinates from 0 to 1000."
"Use the format <point coords=\"X Y\"/>. "
"Return only this tag, nothing else."

Note, that Qwen does not have an internal output convention for pixel pointing, 
therefore the normalized output needs to be forced via the prompt.

## Gemma 3 -L

## RoboPoint -L

# Start pointing task

Start the python scripts via:
`python3 src/molmo2.py`
`python3 src/qwen3-vl.py`

Important: Make sure that the required transformers version is active before starting scripts.

Note, that this will download the model weights locally.
Adjust target picture etc. directly in the script.

# Start Evaluation

Start the evaluation script via:
`python3 src/evaluation.py -name {string} -x {int} -y {int}`

This call requires three arguments:
* -name: Name of the model used
* -x: x-coordinates found by the models, in px
* -y: y-coordinates found by the models, in px

Important: the reference/ground truth image, including the target pixel-coordinates need to be set manually!
The src/groundtruth.py script can be used to generate these target coordinates.
