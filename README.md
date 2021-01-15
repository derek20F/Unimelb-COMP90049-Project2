This file describe my code implement.

There are four python file:

- proj2_chenanf-Baseline.py
- proj2_chenanf-MLP-onehot.py
- proj2_chenanf-MLP-tfidf.py
- proj2_chenanf-NB.py

---

## proj2_chenanf-Baseline.py

In this file, I implemented Zero-R as my baseline model.

---

## proj2_chenanf-MLP-onehot.py

In this file, I implement MLP with One-hot encoding for text type attributes.
MLP parameters could be easily adjust in this program.

After training, this program will plot the accuracy curve of training and validation, and point out the max accuracy of validation on the figure.
Then output the predicted labels of test data set.

---

## proj2_chenanf-MLP-tfidf.py

This file implements MLP with TF-IDF encoding for "tag" and "title" attributes.
MLP parameters could be easily adjust in this program.

After training, this program will plot the accuracy curve of training and validation, and point out the max accuracy of validation on the figure.
Then output the predicted labels of test data set.

To decide which attributes to be used for training, you can do it by comment out some of the code. (There are some hint inside this python file)

If you only want to use the **audio and visual** data for training:

- comment out line 109-114, 124-134
- uncomment line 144-149

If you only want to use **tag** attributes for training:

- comment out line 109-114, 125-128, 144-146
- uncomment line 131-134, 147-149

If you only want to use **title** attributes for training:

- comment out line 109-114, 131-134, 147-149
- uncomment line 125-128, 144-146

If you want to use **tag**, **title** and **visual, audio** attributes

- comment out line 109-114, 144-149
- uncomment line 125-134



---

## proj2_chenanf-NB.py

This file implements the Naive Bayes model for classification. You can switch which NB model to use inside this python file. (line 237-241)

Also, in this file, functions for One-hot and TF-IDF encoding are provided.

If you only want to use the **audio and visual** data for training:

- comment out line 191, 192, 202-205
- uncomment line 215, 216

If you only want to use **tag** attributes for training:

- uncomment line 191,192 202-205
- comment line 215, 216

If you want to use both **tag** and **visual, audio** attributes

- comment out line 202-205, 215, 216
- uncomment line 191, 192

---



