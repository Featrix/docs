Usage
=====

.. meta::
   :description: Using the Featrix client API for creating data embeddings. 
   :keywords: featrix, featrixclient, python, pytorch, ml, ai

.. highlight:: python
    :linenothreshold: 3


.. _installation:

Installation
------------

To use Featrix, first install the client using pip:

.. code-block:: console

   $ pip install featrix-client     # Coming soon. 


You'll also need a Featrix server; you can run the enterprise edition on-site in your environment or use our hosted SaaS.


What's Included
---------------

The ``featrix-client`` package includes a few key modules:

+-------------------+-----------------------------------------------------------+
| ``networkclient`` | A `FeatrixTransformerClient` for                          |
|                   | accessing a Featrix embedding service.                    |
+-------------------+-----------------------------------------------------------+
| ``graphics``      | A set of functions for plotting embedding similarity.     |
++------------------+-----------------------------------------------------------+
| ``utils``         | A set of functions for working with data that we have     |
|                   | found to be useful.                                       |
+-------------------+-----------------------------------------------------------+

Working with Data
-----------------


.. code-block:: python

    import featrixclient as ft
    import pandas as pd
    df = pd.read_csv(path_to_your_file)

Train a vector space and a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can train multiple models on a single vector space.

Check out our `live Google Colab demo notebooks <https://featrix.ai/demo>` for examples. The general approach is as follows:


.. code-block:: python

    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.25)

    # Connect to the Featrix server. This can be deployed on prem with Docker
    # or Featrix’s public cloud.
    featrix = ft.Featrix("http://embedding.featrix.com:8080")

    # Here we create a new vector space and train it on the data.
    vector_space_id = featrix.EZ_NewVectorSpace(df_train)

    # We can create multiple models within a single vector space.
    # This lets us re-use representations for different predictions
    # without retraining the vector space.
    # Note, too, that you could train the model on a different training
    # set than the vector space, if you want to zero in on something
    # for a specific model.
    model_id = featrix.EZ_NewModel(vector_space_id, 
                                   "Target_column",
                                    df_train)

    # Run predictions
    result = featrix.EZ_PredictionOnDataFrame(vector_space_id,
                                              Model_id,
                                              "Target_column",
                                              df_test)

    # Now result is a list of classifications in the same symbols 
    # as the target column



