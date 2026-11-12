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

* MILESTONE 4: visualize data/write paper

# Prerquisites and Installation

Versions used:
* Ubuntu-26.04
* Python 3.10.22

## PyTorch

Source: https://pytorch.org/get-started/locally/

Install:
`pip3 install torch torchvision`

## Molmo -S

Source: https://github.com/allenai/molmo

Model: Molmo 7B-D

## Qwen3-VL -S

## Gemma 3 -L

## RoboPoint -L
