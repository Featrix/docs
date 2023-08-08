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

To use Featrix, first install it using pip:

.. code-block:: console

   $ pip install featrix-client 


We provide live demo notebooks with real data; please email mitch@featrix.ai if you'd like to check them out. We will post them publicly as we make more progress.



What's Included
---------------

The ``featrix-client`` package includes a few key modules:

+-------------------+-----------------------------------------------------------+
| ``embedit``       | A set of classes for encoding and decoding embeddings.    |
+-------------------+-----------------------------------------------------------+
| ``networkclient`` | A `FeatrixTransformerClient` for                          |
|                   | accessing a Featrix embedding service.                    |
+-------------------+-----------------------------------------------------------+
| ``graphics``      | A set of functions for plotting embedding similarity.     |
+-------------------+-----------------------------------------------------------+



Working with Data
-----------------



.. code-block:: python

    import featrixclient as ft
    import pandas as pd
    df = pd.read_csv(path or url)

Send the data to Featrix
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    df = ... # get a Pandas dataframe from somewhere, e.g., read_csv.

    try:
        featrixClient = ft.FeatrixTransformerClient(protocol="http",
                                                    host="embedding.featrix.com",
                                                    port="8080")

        sendResult = featrixClient.send_data(df)
    except FeatrixDataException, e:
        print("Error:", e)


Get the embeddings back from Featrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    try:
        embeddings, encoders = featrixClient.wait_for_embeddings(sendResult,
                                                                 max_timeout=1200)
    except FeatrixDataException, e:
        print("Error:", e)


At this point, `embeddings` and `encoders` will contain the representations for the data.




..
    Creating recipes
    ----------------

    To retrieve a list of random ingredients,
    you can use the ``lumache.get_random_ingredients()`` function:

    .. autofunction:: lumache.get_random_ingredients

    The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
    or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
    will raise an exception.

    .. autoexception:: lumache.InvalidKindError

    For example:

    >>> import lumache
    >>> lumache.get_random_ingredients()
    ['shells', 'gorgonzola', 'parsley']

