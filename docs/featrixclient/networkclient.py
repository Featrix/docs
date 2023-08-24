#
#  Copyright (c) 2023, Featrix, Inc. All rights reserved.
#
#  Proprietary and Confidential.  Unauthorized use, copying or dissemination
#  of these materials is strictly prohibited.
#

import base64
import json
import pickle
import logging
import socket
import time
import traceback
import uuid

import pandas as pd
import requests

logger = logging.getLogger("featrix-client")
# logger.basicConfig(format='%(asctime)s %(message)s', level=logger.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("featrix-client.log")
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter("featrix %(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


__version__ = "<no version information available>"


def __init_version():
    try:
        from .PublishTime import PUBLISH_HOST, PUBLISH_TIME

        global __version__
        __version__ = f"published at {PUBLISH_TIME} from {PUBLISH_HOST}"
    except Exception as e:
        __version__ = f"error: {e}"
    return


__init_version()


class Featrix:
    def __init__(
        self,
        url="http://embedding.featrix.com:8080"
    ):
        """
        Create a Featrix client object.

        Parameters
        ----------
        url : str
            The url of the Featrix server you are using.

            The default is http://embedding.featrix.com:8080

        Returns
        -------
        A Featrix object ready to embed!
        """
        pass

    def EZ_NewVectorSpace(self, df: pd.DataFrame):
        """
        Create a new vector space on a dataframe. The dataframe should include all target columns that you might want to predict.

        You do not need to clean nulls or make the data numeric; pass in strings or missing values all that you need to.

        You can pass in timestamps as a string in a variety of formats (ctime, ISO 8601, etc) and Featrix will detect and extract features as appropriate.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with both the input and target values.
            This can be the same as the dataframe used to train models in vector space.

        Returns
        -------
        str uuid of the vector space (which we call vector_space_id on other calls)

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:
 
        .. hightlight:: python
        .. code-block:: python
            
            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values

        And pass the all_df to `EZ_NewVectorSpace`
        """
        pass

    def EZ_NewModel(self,
                    vector_space_id:str,
                    target_column_name:str,
                    df:pd.DataFrame):
        """
        Create a new model in a given vector space.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`
        target_column_name : str
            Name of the target column. Needs to be present in the passed DataFrame `df`
        df : pd.DataFrame
            The dataframe with both the input and target values. This can be the same as the dataframe used to train the vector space.

        Returns
        -------
        str uuid of the model (which we call model_id on other calls)

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:
        
        .. hightlight:: python
        .. code-block:: python

            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values
        
        And pass the all_df to `EZ_NewModel`
        """
        pass

    def EZ_Prediction(self, vector_space_id, model_id, query):
        """
        Predict a probability distribution on a given model in a vector space.

        Query can be a list of dictionaries or a dictionary for a single query.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`
        model_id : str
            The string uuid name of the model from `EZ_NewModel()`
        query : dict or [dict]
            Either a single parameter or a list of parameters.
            { col1: <value> }, { col2: <value> }

        Returns
        -------
        A dictionary of values of the model's target_column and the probability of each of those values occuring.

        Notes
        -----
        In this version of the API, queries are for categorical values only.
        Scalar querying is coming.
        """
        pass

    def EZ_PredictionOnDataFrame(
        self,
        vector_space_id,
        model_id,
        target_column,
        df: pd.DataFrame,
        check_accuracy=False,
        print_info=False,
    ):
        """
        Given a dataframe, treat the rows as queries and run the given model to provide a prediction on the target
        specified when creating the model.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`
        model_id : str
            The string uuid name of the model from `EZ_NewModel()`
        target_column: str
            The target to remove from the dataframe, if it is present.
        df: pd.DataFrame
            The dataframe to run our queries on.
        check_accuracy: bool, default False
            If True, will compare the result value from the model with the target values from the passed dataframe.
        print_info: bool, default False
            If True, will print out some stats as queries are batched and processed.

        Returns
        -------
        A list of predictions in the symbols of the original target.

        Notes
        -----
        In this version of the API, queries are for categorical values only.
        Scalar querying is coming.
        """
        pass


    def EZ_VectorSpaceEmbeddedColumns(self, vector_space_id):
        """
        Given a vector space id `vector_id`, retrieve the list of columns that were embedded in the vector space.

        If Featrix was unable to process a column, then it will not be in the list.

        Parameters
        ----------
        vector_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`

        Returns
        -------
        A list of column names
        """
        pass


    def EZ_EmbeddingsDistance(self, vector_space_id, col1, col2):
        """
        Given two columns in a vector space, get the cosine distance between their embeddings for
        plotting a heatmap.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`
        col1 : str
            Name of the first column
        col2 : str
            Name of the second column. Can be the same name as `col1` for comparing the column to itself.

        Returns
        -------
        a tuple containing:
            Similarity matrix - the cosine difference of the embeddings to each other.
            col1_members - labels for the col1 dimension of the similarity matrix
            col2_members - labels for the col2 dimension of the similarity matrix

        Notes
        -----
        We provide plot helper functions for matplotlib in graphics.py in this package.
        """
        pass


