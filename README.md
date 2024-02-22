# Hybrid Explanatory Interactive Machine Learning

This repository contains supplementary code to our paper with its full reference:

---

Emanuel Slany, Stephan Scheele, and Ute Schmid (2024):
Hybrid Explanatory Interactive Machine Learning for Medical Diagnosis.

---

In sensitive domains such as medicine, machine learning faces two cruical 
requirements: Experts - in this case physicians - must be able to influence
the model during the optimization. And it must be guaranteed that the
experts' feedback persists for similar instances.
Our method satisfies both requirements
by combining the state-of-the-art CAIPI algorithm with probabilistic
logic rule learning.

### Quick start

`$ pip install -r requirements.txt`

`$ python3 -m scripts.main`

Executing the prior commands install the project's requirements
and start a hyXIML optimization cycle. Hyper-parameter descriptions
can be seen by:

`$ python3 -m scripts.main -h`

### Reproduction of paper experiments

Our paper experiments can be reproduced as follows:

`$ python3 -m scripts.main`

`$  python3 -m scripts.generate_theories`

`$ python3 -m scripts.caipi_hximl_comparison`

Please use the hyper-parameters from the referenced article.
The first command initiates the logging directories,
the second generates the theories (if not already done by the first),
and the final command compares ML to logical inferences - or, CAIPI to hyXIML.

Finally, the command

`$ python3 -m scripts.apply_hximl_from_logging`

executes hyXIML directly from the logging directory,
using existing model and theories.

