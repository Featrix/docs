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

import sys

from unittest import mock

# Mock open3d because it fails to build in readthedocs
#MOCK_MODULES = ["pd"]
#for mod_name in MOCK_MODULES:
#    sys.modules[mod_name] = mock.Mock()

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




class FieldJoiner(object):
    def __init__(self, fields=[], join_op=None):
        assert join_op is not None
        assert join_op == "concat_space", "Only 'concat_space' is supported right now."
        self.fields = fields
        self.join_op = None

    def to_dict(self):
        return {"type": "join_op",
                "fields": self.fields,
                "join_op": self.join_op }

class FieldMapping(object):
    """
    A mapping that links the data from one field in a data set to another field in another data set.

    The purpose is not to join or identify but rather whether the fields map to the same real-life entities
    and thus these values belong in the same place in the vector space.

    This object and the `TableMappings` below can be manually specified or autodiscovered, or a combination of both.
    """
    def __init__(self,
                 target_field:str,
                 source_field: str=None,
                 source_combo: FieldJoiner=None):
        self.target_field = target_field
        self.source_field = source_field

        if source_combo is not None:
            assert isinstance(source_combo, FieldJoiner)
        self.source_combo = source_combo

        if source_field is None and source_combo is None:
            assert "Must specify one of `source_combo` or `source_field`"

    def to_dict(self):
        if self.source_combo is None:
            return {"target_field": self.target_field,
                    "source_field": self.source_field }

        return {"target_field": self.target_field,
                "source_combo": self.source_combo.to_dict() }

class TableMappings(object):
    """
    The full mapping between two tables, compromising a list of one or more field mappings.
    """
    def __init__(self, target:str, source:str, fields:[FieldMapping]):
        self.target = target
        self.source = source

        self.fieldsList = fields

    def verifyOnDataFrames(self, target_df, source_df):
        """
        Ensure the mappings specified line up with the actual data.
        """
        print("verify here")
        return

    def to_dict(self):
        fields = []
        for f in self.fieldsList:
            fields.append(f.to_dict())
        return {"target": self.target,
                "source": self.source,
                "fields": fields }



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

    def EZ_NewVectorSpace(self,
                          df: pd.DataFrame = None,
                          csv_path: str = None,
                          vector_space_id:str = None,
                          on_bad_lines: str = 'skip',
                          ignore_cols=None
                          ):
        """
        Create a new vector space on a dataframe. The dataframe should include
        all target columns that you might want to predict.

        You do not need to clean nulls or make the data numeric; pass in strings or missing values
        all that you need to.

        You can pass in timestamps as a string in a variety of formats (ctime, ISO 8601, etc) and
        Featrix will detect and extract features as appropriate.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with both the input and target values.
            This can be the same as the dataframe used to train models in vector space.

        csv_path: str
            Path to a CSV file to open and read.

        vector_space_id: str
            An id for the vector space. If none is specified, a uuid is used.

        on_bad_lines:
            For reading the CSV file. 'skip', 'error', 'warn'.

        ignore_cols:
            List of columns to ignore when training. If a column is specified that is not found in
            the dataframe, an exception is raised.

        Returns
        -------
        str uuid of the vector space (which we call vector_space_id on other calls)

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:

            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values

        And pass the all_df to `EZ_NewVectorSpace`
        """
        pass

    def EZ_VectorSpaceMetaData(self, vector_space_id):
        """
        Get metadata for the specified vector space.

        Parameters
        ----------
        vector_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`

        Returns
        -------
        A dictionary of metadata. The dictionary contains:
            Information about the columns used to train the vector space.
            The training time, batch dimensions, and other statistics.
            Which encoders were used for which columns and the detected probability of data types.
            Information about every set of training done on the vector space, including loss per iteration and number of epochs.

        The specific format of the dictionary may change from release to release as we improve the product.

        Returns None if the vector space does not exist.
        """
        pass

    def EZ_VectorSpaceEmbeddedColumns(self, vector_space_id):
        """
        Given a vector space id `vector_space_id`, retrieve the list of columns that were embedded in the vector space.

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

    def EZ_EmbedRecords(self,
                        vector_space_id: str,
                        records=None,
                        #                        dictList: [dict] = None,
                        colList: [list] = None):
        """
        Embed new records. You can use this to test embeddings for new data that isn't trained in the vector space
        (or that is); you can pass partial records, the sky is the limit.

        This does not edit the vector space.

        Parameters
        ----------
        vector_space_id: str
            The vector space to use.

        records:
            This can be a dataframe or a list of dictionaries.

            The keys in the dictionary need to be column names from the embedding space (which you can query with
            `EZ_VectorSpaceEmbeddedColumns()`.

        colList:
            A list of keys. You can use this to pass only some of the fields in the records argument without
            having to manually drop or reduce the data.
        """
        pass


    def EZ_EmbeddingsDistance(self, vector_space_id, col1, col2, col1_range=None, col2_range=None, col1_steps=None, col2_steps=None):
        """
        Given two columns in a vector space, get the cosine distance between their embeddings for
        plotting a heatmap.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()` or `EZ_DataSpaceNewVectorSpace()`
        col1 : str
            Name of the first column
        col2 : str
            Name of the second column. Can be the same name as `col1` for comparing the column to itself.
        col1_range:
            optional, tuple of min and max values. (Scalars only)
        col2_range:
            optional, tuple of min and max values. (Scalars only)
        col1_steps:
            optional, number of steps to sample within col1_range. (Scalars only)
        col2_steps:
            optional, number of steps to sample within col2_range. (Scalars only)

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

    def EZ_RemoveVectorSpace(self, vector_space_id: str, force:bool = False):
        """
        Removes a vector space, if it exists. No error if it does not exist.

        Parameters
        ----------
        vector_space_id: str
            Id of the vector space.

        force: bool
            If set to True, will kill a training process if the vector space is currently training.

        """

    def EZ_VectorSpaceNearestNeighbors(self,
                                       vector_space_id:str,
                                       query: dict,
                                       num:int,
                                       database_id = None):
        """
        Given a vector space id `vector_space_id, query the passed vector for the nearest neighbors.

        The neighbors come from the training space by default, though you can create additional vector databases
        for the same vector space with other calls.

        If the columns are specified, then only those columns are considered for the query and search.

        Cannot use `database_id` AND `columns` in the same call.
        """

    def EZ_VectorSpaceCluster(self,
                              vector_space_id:str,
                              k:int,
                              colList: [str] = None,
                              forceNewCluster:bool=False):
        """
        Cluster the training data.

        Future versions of the API will let you cluster new data sets, with and without the training data.

        This call will block until the clustering is complete; this may take a few seconds or minutes.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()`

        k: int
            The number of clusters. There is an art to picking k and we will add tools later
            to help. If you pick k to be too small, many clusters will appear to be similar.

            Note that this implementation does *not* split the data into equal sized k groups.

        colList: [str]
            The column names to cluster on.

            NOTE: The current release supports just one column. This will be fixed ASAP.

            This defaults to None. When None is used, the cluster index will be built considering
            all columns. Often times we want to experiment with different arrangements and this
            lets us do that--we can verify clusters on a single column or a reduced subset
            without training a new vector space.

        Returns
        -------
        A dictionary with a few values:

        {
            finished: True - indicates the clustering process has finished.
            error: if present, it will be set to True, and indicate something has gone wrong.
            message: error message if `error` is set.

            result: The result dictionary if there was no error:
                {
                    centroids: The vector centers of each cluster.
                    id_map: A dictionary that maps the cluster offsets to the original data file. The values in the dictionary contain the file name, hash of the data, and row index. Helper functions to deal with this are coming in a future release.
                    label_histogram: A convenience histogram of k elements that indicates the number of items in each cluster.
                    labels: Similar to sklearn's fit() results, this maps offsets (which are keys into id_map) to the cluster id.
                    total_square_error: The total squared error in each cluster (contains k elements)
                }
        }
        """
        pass

    def EZ_ClusterLabelsToClusterDict(self, labels, df=None):
        """
        Given the labels in `EZ_VectorSpaceCluster` (see result['labels']),
        build a dictionary of the labels and which indexes were in the data.

        If the dataframe is passed, then the indexes are mapped in and the
        data is returned instead.
        """
        pass

    def EZ_DataSpaceExists(self, data_space_name:str):
        """
        Returns True if the data space was confirmed to exist.
        """
        pass

    def EZ_DataSpaceMetaData(self, data_space_name:str):
        """
        Retrieve the metadata for the specified data space.
        """
        pass

    def EZ_DataSpaceCreate(self, data_space_name:str, metadata:dict=None):
        """
        Creates a new data space to be used to train downstream vector spaces.

        You can load your own metadata into the passed dictionary for tracking any metadata you need.

        Parameters
        ----------
        data_space_name : str
            Arbitrary name you want to use to refer to the data space.

        metadata: dict
            Arbitrary metadata for version, debug capture, whatever you need.
            If None, a metadata dict with the creation time will be supplied for you.

        Returns
        -------
        A handle confirming the name or an error if the name is already in use.
        """
        pass

    def EZ_DataSpaceLoadIfNeeded(self,
                                 data_space_id:str,
                                 path: str=None,
                                 label: str=None,
                                 df: pd.DataFrame=None,
                                 on_bad_lines: str='warn',
                                 sample_percentage: float=None,
                                 sample_row_count: int=None):
        """
        Copy a file or dataframe to the Featrix server if the file does not exist

        Also associates the file with the specified data space.

        The file can be associated with multiple dataspaces.

        Safe to call this multiple times.

        Parameters
        ----------
        data_space_id : str
            the name of the data space.

        path : str
            either use this for the dataframe `df` but not both.

        label : str
            this is the label for the file that will be used in this data space.

        on_bad_lines: str
            this is passed to pandas pd.read_csv without editing. 'skip' will ignore the bad lines; 'error' will stop loading and fail if there are bad lines. In the current software, passing 'warn' will not get returned to the API client (we need to fix this).

        sample_percentage: float.
            Take a percentage of the rows at random for training the vector. The sample will be captured at the time
            the vector space is trained; in other words, which part of the data is sampled will change on every training.

        sample_row_count: int
            Take an absolute number of rows. Cannot be used with `sample_percentage`.

        Notes
        -----
        Files are compared with a local md5 hash and a remote md5 hash before deciding to transmit the file.
        No partial copies are supported.
        """
        pass

    def EZ_DataSpaceAutoJoin(self, data_space_id: str):
        """
        Computes auto joining possibilities for the specified data space.

        This lets us investigate the linking performed with `EZ_DataSpaceNewVectorSpace()` so that we
        can review it and tweak it if needed. The resulting dictionaries can be modified and passed to
        `EZ_DataSpaceNewVectorSpace` for training. You can also pass the mappings to `EZ_DataSpaceDoProjection`
        to see example projections from the mappings.

        There are two goals with the mappings:

        First, we want to map fields that uniquely identify objects to whatever degree we can.

        Second, we want to map mutual information that spans data sets so that those fields
        get input into the same place in our input vectors to the embeddings transformation.

        We infer the second set of mappings by leveraging the first set to identify linked records
        and then we sample and look for high conditional probabilities of fields resulting in the
        same field. For example, in this dataset, the joint distribution of unrelated fields often
        looks promising, such as when comparing "building square feet" and "street number" from
        different tables. But when we condition this comparison to specific linked entities,
        those unrelated false positives no longer hold and we latch onto the "correct" associations,
        such as zip code information across different source files, even if their unconditioned
        mutual information was "closer".

        This feature is in beta.

        This can be used to confirm there are statistical relationships that are clearly
        present in the various files associated. This serves to diagnose and verify behavior
        before training the vector space on the data.

        You can also get back a CSV file of the projection to examine before creating
        the vector space.

        The linkage includes a list of the full columns in the result.

        The columns get a hierarchical naming:

        The base data columns are not changed.
        Additional data files get the label as a prefix with all spaces converted to underscores, and an underscore before the field name.

        Parameters
        ----------
        data_space_id: str
            the id returned from `EZ_DataSpaceCreate`

        Returns
        -------
        Dictionary associating each file in the dataspace and the best detected linking columns.
        """
        pass

    def EZ_DataSpaceSetMappings(self, dataspace_name, mappings:TableMappings):
        """
        Set the mappings between the base data set and another data set in the dataspace.

        This overwrites any previously set mapping for the two specified data sets, but
        does not change the relationships between other pairs in the dataspace.
        """
        pass

    def EZ_DataSpaceSetIgnoreColumns(self, dataspace_name, ignore_list:[str]):
        """
        Set a list of columns to ignore. If any are set, this will overwrite the list.

        The column names specified are the final projected names.
        """

    def EZ_DataSpaceNewVectorSpace(self,
                                   data_space_id: str,
                                   vector_name: str = None,
                                   metadata: dict = None,
                                   ignore_cols: [str] = None,
                                   detect_only: bool = False
                                   ):
        """
        Create a multimodal vector space on data space.

        This lets you create multimodal embeddings for assorted tabular data sources in a single
        trained vector space. This essentially lets you build a foundational model on your data
        in such a way that you can query the entire data by using partial information that maps into
        as little as one of your original data sources, or you can leverage partial information
        spanning multiple data sources.

        You can create multiple vector spaces from a data space; you can use a subset of the data,
        ignore columns, or change mappings to rapidly experiment with models using this call.

        The data space must already be loaded with 1 or more data source files.

        This function will use auto-join (which you can try directly with `EZ_DataSpaceAutoJoin`)
        to find the linkage and corresponding overlapping mutual information between data files
        that have been loaded. Then a new vector space is trained with the following columns:

            Base data file: all columns (unless ignored in the ignore_cols parameter)

            2nd data file:  all columns, renamed to <2nd data file label> + "_" + <original_col_name>
                            However, the columns used for linking will not be present, as they
                            will get their mapped names in the base data file.

                            To ignore a column in the 2nd data file, specify the name in the
                            transformed format.

            3rd data file:  same as 2nd data file.

        This trains the vector space in the following manner:

            Let's imagine the 2nd_file_col1 and 3rd_file_col2 are the linkage to col1 in the base
            data set. The training space will effectively be a sparse matrix:

                col1                    col2          col3        2nd_file_col2       2nd_file_col3       3rd_file_col2
                values from base data.....................        [nulls]                                 [nulls]
                .
                .
                .
                2nd_file_col1 in col1   [nulls]                   values from 2nd file................... [nulls]
                .                       .                         .
                .                       .                         .
                3rd_file_col1 in col2   [nulls]                   [nulls]                                 values from 3rd file
                .                       .                         .                                       .
                .                       .                         .                                       .
                .                       .                         .                                       .

        Params
        ------
        data_space_id: str
            the id returned from `EZ_DataSpaceCreate`

        vector_name: str
            You can specify your own name for the vector space, or Featrix will assign one for you.

        metadata: dict
            Your own dictionary of metadata if you want to track version information or other characteristics.

        ignore_cols:
            A list of columns you want to ignore. You can get the list of columns, which may include renamed columns
            from merging multiple files, by calling detect_only=True

        detect_only:
            If set to True, Featrix will not create the vector space, but instead it will construct the mapping links,
            detect data types and encoders, identify opportunities for enrichment, and return all of this information
            to you.

        mapping:
            By default, Featrix will establish links between files in the data space and use these links to construct
            the mapping for the vector space. If Featrix cannot automatically map the data, an error will be returned.
            You can override the Featrix mapping behavior using this parameter to pass a dictionary of mappings.

            This dict is in the format returned by `EZ_DataSpaceAutoJoin`.

        """
        pass

    def EZ_NewModel(self,
                    vector_space_id:str,
                    target_column_name:str,
                    df:pd.DataFrame,
                    modelSize:str = 'auto'):
        """
        Create a new model in a given vector space.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()` or `EZ_DataSpaceNewVectorSpace()`
        target_column_name : str
            Name of the target column. Needs to be present in the passed DataFrame `df`
        df : pd.DataFrame
            The dataframe with both the input and target values. This can be the same as the dataframe
            used to train the vector space.
        epochs: int
            Number of epochs to train on.
        modelSize: str
            Can be 'auto', 'small', 'large'
            For models that run in the Featrix server, 'small' is a 2 hidden layer model with 25 dimensions.
            'large' is a 6 layer model with 50 dimensions.

        Returns
        -------
        str uuid of the model (which we call model_id on other calls)

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:

            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values

        And pass the all_df to `EZ_NewModel`
        """
        pass

    def EZ_DetectEncoders(self,
                          df: pd.DataFrame = None,
                          csv_path: str = None,
                          on_bad_lines: str = 'skip',
                          print_result: bool = True):
        """
        Query Featrix with some data to see how Featrix will interpret the data for encoding
        into the vector embeddings. You can override these detections if needed.

        This will also return information about enriched columns that Featrix will extract.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with both the input and target values.
            This can be the same as the dataframe used to train models in vector space.

        csv_path: str
            Path to a CSV file to open and read.

        on_bad_lines:
            For reading the CSV file. 'skip', 'error', 'warn'.

        print_result:
            If True, prints the result to the console as a nice table. Same as calling `EZ_PrintDetectOnly` on the returned result.

        Returns
        -------
        A dictionary of the printed table; useful for storing or comparing results across
        data sets or different runs. You can ignore this if you are using `print_result=True`.
        """
        pass

    def EZ_PrintDetectOnly(self, metadict):
        """
        Prints the metadata when detecting the encoders for training a vector space

        Nothing is returned.
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
        A dictionary of values of the model's target_column and the probability of each of those values occurring.
        """
        pass

    def EZ_PredictionOnDataFrame(
        self,
        vector_space_id,
        model_id,
        df: pd.DataFrame,
        target_column: str=None,
        include_probabilities: bool=False,
        check_accuracy: bool=False,
        print_info: bool=False,
    ):
        """
        Given a dataframe, treat the rows as queries and run the given model to provide a prediction on the target
        specified when creating the model.

        Parameters
        ----------
        vector_space_id : str
            The string uuid name of the vector space from `EZ_NewVectorSpace()` or `EZ_DataSpaceNewVectorSpace()`
        model_id : str
            The string uuid name of the model from `EZ_NewModel()`
        target_column: str
            The target to remove from the dataframe, if it is present.
            `None` will default to the target column of the model.
        df: pd.DataFrame
            The dataframe to run our queries on.
        include_probabilities: bool, default False
            If True, the result will be a list of dictionaries of probabilities of values.
            This works like sklearn's `predict_proba()` on classifiers, though our return value is not an ndarray.
            If `check_accuracy` is set to true, this will just ensure that the highest probability is right; we do not
            (yet) support checking an ordered list of probabilities or other nice things like that.
        check_accuracy: bool, default False
            If True, will compare the result value from the model with the target values from the passed dataframe.
        print_info: bool, default False
            If True, will print out some stats as queries are batched and processed.

        Returns
        -------
        A list of predictions in the symbols of the original target.
        """

        pass


