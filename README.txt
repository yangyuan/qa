Download glove.840B.300d.txt in data folder.
https://nlp.stanford.edu/projects/glove/
(make sure you can see data/glove.840B.300d.txt)

Download and unzip bAbI data to data/babi/tasks_1-20_v1-2
https://research.fb.com/downloads/babi/
(make sure you can see data/babi/tasks_1-20_v1-2/LICENSE.txt)

Download and unzip MACRO data to data/macro
http://www.msmarco.org/dataset.aspx
(make sure you can see data/macro/dev_v1.1.json)

Download and unzip SQuAD data to data/squad
https://rajpurkar.github.io/SQuAD-explorer/
(make sure you can see data/squad/dev-v1.1.json)

>>> How to process data:
Run preprocess.py
Which will take few minutes.

>>> How to run demo:
Make sure data/model/_general/ has checkpoint files and model files.
Run service.py
Visit demo http://127.0.0.1/index.html

>>> How to train your own model:
Run model.py
Models will be in data/model/batch and data/model/epoch folders


WARNING: 10-20 epochs are usually required for training.
GPU:
8G graphic memory is required for training this model, one batch need 5-10 seconds.
CPU:
One batch need 2-4 minutes, 1 epoch will need a week.