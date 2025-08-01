# Model class definitions module

## model_base.py

- exports *Model* which defines the basic model of Markus without learning
- serves as a parent class for model extensions

## learning_models.py

- exports extensions with different learning rules
- importantly:
  - *PiLearnModelUdelay* as standard gradient descent learner
  - *PiLearnModelUdelayProb* as Bayesian learner
- compare docstrings

## feedback_models.py

- exports model extensions with dynamic $g, \Pi, \mu$ through feedback
- furtermore exports class *GFeedbackExtension*
  -  useable through static method *GFeedbackExtension.make* that takes another model class
     as well as initiazation parameters that creates an instance of that model extented
     by a feedback dynamic on $g$
