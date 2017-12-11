(1) what and how you addressed the feedback your team got for the presentation
(both are mentioned in final report)

Feedback 1: It would be nice if you can compare the outputs from different systems.

Among the three main data sets, SQuAD is the most popular data set. Our score can be listed between 40-50 on SQuAD
Leader board. https://rajpurkar.github.io/SQuAD-explorer/
But please notice that SQuAD is very biased on short answers. But when training our own model, the real-world data
MS MACRO is the largest. So our model MIGHT performs better on real world questions.

Feedback 2: Discuss why the data preprocessing is done in this way (removing long sentences).

To clarify, we didn't remove long sentences, we remove long passages, along with it's questions/answers.
Two main reasons:
1. To save time. We have sufficient data, the training time is long enough, it's okay to have less data.
2. If we crop the passage, the answers will be more likely in the center of the passage, which make the data biased.


(2) what changes your team made after the presentation.

We refactored our code so that it can be more readable and extendable.
(Due to the refactoring) We trained a new instance with SQuAD/MACRO data sets only, which gives similar performance.
We removed the auto-pause (when overfitting) feature, because it turns out to be unstable, instead, we keep old models
and record the performance (with STDOUT redirecting).