# tree-kernel

Code of a baseline method to automatically detect disease-chemical relationships in biomedical papers.

The method works by computing a word embedding of the training corpus, concatenating the embeddings of disease-chemical pairs (into one vector of ~100 dimensions), to train a SVM with a quadratic kernel. Tree kernels were also tried, but their impact on classification was negative (compared to embeddings or bags of words).

As training and test corpus we use the known CDR corpus (BioCreative). The baseline has an accuracy of 80%. The whole experiment is self-contained. Download and type on the command line (in the package directory):

```
   python main.py
```

to run the experiment.
