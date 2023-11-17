API
===

.. toctree::

.. autosummary::


The best way to use Featrix is via the object API provided in Python.

Working with it is pretty easy. You can call help() on Python objects to see the docstrings on the objects.

Generally, you create a Featrix object for a specific server.

The API contains three primary objects: `FeatrixEmbeddingSpace`, `FeatrixModel`, and `FeatrixDataSpace`.

The `FeatrixDataSpace` is a way to associate data into one or more embedding spaces. Using this is required when you want to train an embedding space using multiple sources of data together.


The `FeatrixEmbeddingSpace` is trained on the contents of a data space, or it can be trained directly on a single table (e.g., a Pandas dataframe or a CSV file).

And finally, the `FeatrixModel` represents a predictive model. It is trained using an embedding space and either new data for the model specifically or some of the data that the embedding space was trained on, depending on your application needs.

You can build scalar predictions, classifications, recommendations, and more with this API. You can also cluster data or query for nearest neighbors by leveraging the embedding space. You can extend the embedding spaces, branch them, tune their training, and more.

We have designed this API to work with standard Python ecosystems: you should be able to easily connect Pandas, databases, matplotlib, sklearn, numpy, and PyTorch with the API. If you run into issues or want enhancements, drop us a note at mitch@featrix.ai or join our `Slack community <https://join.slack.com/t/featrixcommunity/shared_invite/zt-25ad0tj5j-3VyaO3YdI8qI4kdr2VhUGA>`!


.. autoclass::  featrixclient.networkclient.Featrix
    :members:

.. autoclass::  featrixclient.networkclient.FeatrixDataSpace
    :members:

.. autoclass::  featrixclient.networkclient.FeatrixEmbeddingSpace
    :members:

.. autoclass::  featrixclient.networkclient.FeatrixModel
    :members:

