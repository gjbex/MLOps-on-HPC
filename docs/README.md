Machine learning is becoming increasingly important in many fields, and topics
such as reproducibility, data management, and experiment tracking are essential
for any machine learning project. This training will teach you how to use MLOps
frameworks to manage your machine learning projects, data, and workflows on HPC
systems.


## Learning outcomes

When you complete this training you will

  * be able to use DVC to version your data;
  * be able to define pipelines with DVC;
  * be able to use DVC to manage your machine learning projects;
  * be able to reproduce your machine learning experiments;
  * be able to compare experiments using Git, DVC, and DVCLive.


## Schedule

Total duration: 4 hours.

  | Subject                                     | Duration |
  |---------------------------------------------|----------|
  | introduction and motivation                 |  5 min.  |
  | setting up a Git repository for ML          | 15 min.  |
  | versioning data using DVC                   | 30 min.  |
  | defining a workflow with DVC                | 60 min.  |
  | comparing experiments using Git and DVC     | 60 min.  |
  | tracking experiments using DVCLive          | 60 min.  |
  | wrap up                                     | 10 min.  |


## Training materials

Slides are available in the [GitHub
repository](https://github.com/gjbex/MLOps-on-HPC), as well as example code and
hands-on material.


## Target audience

This training is for you if you need to manage machine learning workflows and
experiments on HPC systems.


## Prerequisites

You will need experience running machine learning workloads in Python or R.
You will also need to be comfortable on the command line, and have some
experience using Git version control.

More concretely, participants should already be comfortable with the following:

* running machine learning or data-analysis scripts in Python or R;
* working from the shell: navigating directories, running commands, editing
  small files, and inspecting output;
* using Git for everyday operations such as `clone`, `add`, `commit`,
  `status`, `log`, and checking out previous revisions;
* understanding the difference between source code, data, parameters, and
  generated results in an experiment;
* running jobs on an HPC system at a basic level if the hands-on material is
  executed on a cluster.

You do not need prior experience with DVC, DVCLive, DVC pipelines, experiment
queues, or DVC-based parameter sweeps. Those are part of the training itself.

### Quick self-assessment

If you can do most of the tasks below without looking up basic shell, Git, or
ML-workflow syntax, you are likely ready for this training.

* run a Python or R script that reads input data and writes output files;
* initialize or clone a Git repository and make a small commit;
* inspect which files in a repository changed after running an experiment;
* read a short YAML or configuration file and identify a few parameter values;
* understand that code, data, parameters, metrics, and plots may all change
  between experiments;
* rerun an experiment after changing one parameter and compare the result;
* work on a remote HPC system from the command line.

If several of these items still feel difficult, the training will probably move
too fast. In that case, it is better to first refresh basic command-line use,
Git, and the workflow you use to run machine learning experiments.

For following along hands-on, you need
* laptop or desktop with internet access and set up so you can connect to an
  HPC system;
* an account on an HPC system (e.g., VSC, CECI, ...);
* compute credits if that is required to run jobs on the HPC system;


## Level of the Material

For participants who already have basic machine learning workflow experience,
the material in this training is approximately

* Introductory: 25 %
* Intermediate: 55 %
* Advanced: 20 %

These percentages describe the level of the MLOps and experiment-management
topics covered in the training, not the required entry level in Python, R, or
Git itself.


## Trainer(s)

  * Geert Jan Bex ([geertjan.bex@uhasselt.be](mailto:geertjan.bex@uhasselt.be))
