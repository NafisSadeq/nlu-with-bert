BERTopic uses HDSCAN for clustering the data and it cannot specify the number of clusters you would want. To a certain extent, 
this is an advantage, as we can trust HDBSCAN to be better in finding the number of clusters than we are.
Instead, we can try to reduce the number of topics that have been created. Below, you will find three methods of doing 
so. 
  
### **Manual Topic Reduction**
Each resulting topic has its own 
feature vector constructed from c-TF-IDF. Using those feature vectors, we can find the most similar 
topics and merge them. If we do this iteratively, starting from the least frequent topic, we can reduce the number 
of topics quite easily. We do this until we reach the value of `nr_topics`:  

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics=20)
```

### **Automatic Topic Reduction**
One issue with the approach above is that it will merge topics regardless of whether they are very similar. They 
are simply the most similar out of all options. This can be resolved by reducing the number of topics automatically. 
It will reduce the number of topics, starting from the least frequent topic, as long as it exceeds a minimum 
similarity of 0.915. To use this option, we simply set `nr_topics` to `"auto"`:

```python
from bertopic import BERTopic
topic_model = BERTopic(nr_topics="auto")
```

### **Topic Reduction after Training**
Finally, we can also reduce the number of topics after having trained a BERTopic model. The advantage of doing so, 
is that you can decide the number of topics after knowing how many are created. It is difficult to 
predict before training your model how many topics that are in your documents and how many will be extracted. 
Instead, we can decide afterward how many topics seem realistic:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
 
# Create topics -> Typically over 50 topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

# Further reduce topics
new_topics, new_probs = topic_model.reduce_topics(docs, topics, nr_topics=30)
```

The reasoning for putting `docs` and `topics` (and optionally `probabilities`) as parameters is that these values are not saved within 
BERTopic on purpose. If you were to have a million documents, it is very inefficient to save those in BERTopic 
instead of a dedicated database.  

