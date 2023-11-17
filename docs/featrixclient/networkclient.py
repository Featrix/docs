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


class FeatrixConnectionError(Exception):
    def __init__(self, url, message):
        # logger.error("Connection error for url %s: __%s__" % (url, message))
        super().__init__("Connection error for URL %s: __%s__" % (url, message))


class FeatrixServerResponseParseError(Exception):
    def __init__(self, url, payload):
        # logger.error("Error parsing result from url %s: __%s__" % (url, payload))
        super().__init__("Bad response from URL %s: __%s__" % (url, payload))


class FeatrixBadServerCodeError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = "Unexpected status code %s: %s" % (status_code, message)
        super().__init__(self.message)


class FeatrixServerError(Exception):
    def __init__(self, msg):
        self.message = msg
        super().__init__(self.message)


class FeatrixEmbeddingSpaceNotFound(FeatrixBadServerCodeError):
    """
    Embedding space not found on server.
    """
    def __init__(self, vector_space_id):
        self.message = f"Embedding space \"{vector_space_id}\" not found."
        super().__init__(400, self.message)

class FeatrixDataSpaceNotFound(FeatrixBadServerCodeError):
    """
    Data space not found on server.
    """
    def __init__(self, data_space):
        self.message = f"Data space \"{data_space}\" not found."
        super().__init__(400, self.message)

class FeatrixModelNotFound(FeatrixBadServerCodeError):
    """
    Model not found in embedding space.
    """
    def __init__(self, model_id, vector_space_id):
        self.message = f"Model \"{model_id}\" not found in embedding space \"{vector_space_id}\"."
        super().__init__(400, self.message)


class FeatrixDatabaseNotFound(FeatrixBadServerCodeError):
    """
    Specified column not found.
    """
    def __init__(self, db_id, vector_space_id):
        self.message = f"Database \"{db_id}\" not found in embedding space \"{vector_space_id}\"."
        super().__init__(400, self.message)

class FeatrixColumnNotFound(FeatrixBadServerCodeError):
    """
    Specified column not found.
    """
    def __init__(self, vector_space_id, col_name, all_col_names):
        self.message = f"Column \"{col_name}\" not found in embedding space \"{vector_space_id}\". not found. Available column names: {', '.join(all_col_names)}."
        super().__init__(400, self.message)

class FeatrixColumnNotAvailable(FeatrixBadServerCodeError):
    """
    Specified column not available. Typically this means we weren't able to encode or decode it.
    """
    def __init__(self, vector_space_id, col_name, encoded_columns):
        self.message = f"Column \"{col_name}\" not available for encoding in embedding space \"{vector_space_id}\". Available columns with codecs: {', '.join(all_col_names)}."
        super().__init__(400, self.message)

class FeatrixInvalidModelQuery(FeatrixBadServerCodeError):
    def __init__(self, p):
        self.message = f"Invalid model query: %s" % p
        super().__init__(400, self.message)


def ParseFeatrixError(s):
    PREFIX = "featrix_exception:"
    if s.startswith(PREFIX):
        s = s[len(PREFIX):]
        try:
            jr = json.loads(s)
        except:
            return Exception("Internal error: Couldn't parse __%s__ into Featrix exception" % s)
        error_name = jr.get('error_name')
        p = jr.get('params')
        if error_name == "vector space not found":
            return FeatrixEmbeddingSpaceNotFound(p.get('vector_space_id'))
        elif error_name == "column not found":
            return FeatrixColumnNotFound(p.get('vector_space_id'),
                                        p.get('col_name'),
                                        p.get('all_col_names'))
        elif error_name == "column not encoded":
            return FeatrixColumnNotAvailable(p.get('vector_space_id'),
                                             p.get('col_name'),
                                             p.get('encoded_col_names'))
        elif error_name == "data space not found":
            return FeatrixDataSpaceNotFound(p.get('data_space_id'))
        elif error_name == "model not found in embedding space":
            return FeatrixModelNotFound(p.get('model_id'),
                                        p.get('vector_space_id'))
        elif error_name == "database not found in embedding space":
            return FeatrixDatabaseNotFound(p.get('db_id'),
                                           p.get('vector_space_id'))
        elif error_name == "invalid model query":
            return FeatrixInvalidModelQuery(p)

    #raise Exception(f"Unexpected error: {s}")
    return None


__version__ = "Featrix APT 1.2"


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
        A Featrix object ready for embedding data.
        """
        pass




class FeatrixDataSpace:
    def __init__(self, client, data_space_name=None):
        self.client = client
        self.data_space_name = data_space_name

    def clearAll(self):
        """
        A shortcut to remove and create a dataspace. Great for ensuring reproducible results.
        """
        self.removeIfExists()
        self.create()
        return

    def create(self, metadata:dict=None):
        """
        Creates a new data space to be used to train downstream embedding spaces.

        You can load your own metadata into the passed dictionary for tracking any metadata you need.

        Parameters
        ----------
        data_space_name : str
            Arbitrary name you want to use to refer to the data space.
            If none, a string of an uuid will be used and returned.

        metadata: dict
            Arbitrary metadata for version, debug capture, whatever you need.
            If None, a metadata dict with the creation time will be supplied for you.

        Returns
        -------
        A handle confirming the name or an error if the name is already in use.
        """
        return self.client.EZ_DataSpaceCreate(self.data_space_name,
                                              metadata=metadata)

    def removeIfExists(self):
        """
        Remove the data space if it exists. Any vector spaces built from the data space are NOT deleted.
        """
        assert self.data_space_name is not None
        return self.client.EZ_DataSpaceRemoveIfNeeded(self.data_space_name)

    def exists(self):
        """
        Returns True if the data space was confirmed to exist.
        """
        return self.client.EZ_DataSpaceExists(data_space_name=self.data_space_name)

    def metadata(self):
        """
        Retrieve the metadata for the specified data space.
        """
        return self.client.EZ_DataSpaceMetaData(data_space_name=self.data_space_name)

    def loadFileIfNeeded(self,
                         path: str = None,
                         label: str = None,
                         df: pd.DataFrame = None,
                         on_bad_lines: str = 'warn',
                         sample_percentage: float = None,
                         sample_row_count: int = None,
                         drop_duplicates: bool = True
                         ):
        """
        Copy a file or dataframe to the Featrix server if the file does not exist

        Also associates the file with the specified data space.

        The file can be associated with multiple dataspaces.

        Safe to call this multiple times.

        Parameters
        ----------
        path : str
            either use this for the dataframe `df` but not both.

        label : str
            this is the label for the file that will be used in this data space.

        df: pd.DataFrame
            use either this or `path`, but not both.

        on_bad_lines: str
            this is passed to pandas pd.read_csv without editing. 'skip' will ignore the bad lines; 'error' will stop
            loading and fail if there are bad lines. In the current software, passing 'warn' will not get returned to
            the API client (we need to fix this).

        sample_percentage: float.
            Take a percentage of the rows at random for training the vector. The sample will be captured at the time
            the embedding space is trained; in other words, which part of the data is sampled will change on every training.

        sample_row_count: int
            Take an absolute number of rows. Cannot be used with `sample_percentage`.

        drop_duplicates: bool
            Ignore duplicate rows. True by default. The dropping will occur *before* sampling.

        Notes
        -----
        Files are compared with a local md5 hash and a remote md5 hash before deciding to transmit the file.

        File hashes happen on the entire file; the data file is not sampled or de-duplicated prior to training a vector
        space. In other words, the sampling and de-duplication parameters are intended for convenience and not to save
        bandwidth or storage. We are open to feedback on this behavior. One implication is that samples will vary across
        trainings.

        No partial copies are supported.
        """
        return self.client.EZ_DataSpaceLoadIfNeeded(data_space_id=self.data_space_name,
                                                    path=path,
                                                    label=label,
                                                    df=df,
                                                    on_bad_lines=on_bad_lines,
                                                    sample_percentage=sample_percentage,
                                                    sample_row_count=sample_row_count,
                                                    drop_duplicates=drop_duplicates)

    def autoJoin(self):
        """
        Computes auto joining possibilities for the specified data space.

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
        before training the embedding space on the data.

        You can also get back a CSV file of the projection to examine before creating
        the embedding space.

        The linkage includes a list of the full columns in the result.

        The columns get a hierarchical naming:

        The base data columns are not changed.
        Additional data files get the label as a prefix with all spaces converted to underscores, and an underscore before the field name.

        Parameters
        ----------
        None

        Returns
        -------
        Dictionary associating each file in the dataspace and the best detected linking columns.
        """
        return self.client.EZ_DataSpaceAutoJoin(self.data_space_name)

    # def verifyMapping(self):
    #     pass

    def setMappings(self, mappings):
        """
        Set the mappings between the base data set and another data set in the dataspace.

        This overwrites any previously set mapping for the two specified data sets, but
        does not change the relationships between other pairs in the dataspace.
        """
        return self.client.EZ_DataSpaceSetMappings(self.data_space_name, mappings)

    def ignoreColumns(self, ignore_list:[str]):
        """
        Set a list of columns to ignore. If any are set, this will overwrite the list.

        The column names specified are the final projected names.
        """
        return self.client.EZ_DataSpaceSetIgnoreColumns(self.data_space_name, ignore_list=ignore_list)

    def newEmbeddingSpace(self,
                          metadata: dict = None,
                          ignore_cols: [str] = None,
                          detect_only: bool = False,
                          n_epochs: int = 5,
                          learning_rate: float = 0.01,
                          print_debug=False
                          ):
        """
        Create a multimodal embedding space on data space.

        This lets you create multimodal embeddings for assorted tabular data sources in a single
        trained embedding space. This essentially lets you build a foundational model on your data
        in such a way that you can query the entire data by using partial information that maps into
        as little as one of your original data sources, or you can leverage partial information
        spanning multiple data sources.

        You can create multiple embedding spaces from a data space; you can use a subset of the data,
        ignore columns, or change mappings to rapidly experiment with models using this call.

        The data space must already be loaded with 1 or more data source files.

        This function will use auto-join (which you can try directly with `EZ_DataSpaceAutoJoin`)
        to find the linkage and corresponding overlapping mutual information between data files
        that have been loaded. Then a new embedding space is trained with the following columns:

            Base data file: all columns (unless ignored in the ignore_cols parameter)

            2nd data file:  all columns, renamed to <2nd data file label> + "_" + <original_col_name>
                            However, the columns used for linking will not be present, as they
                            will get their mapped names in the base data file.

                            To ignore a column in the 2nd data file, specify the name in the
                            transformed format.

            3rd data file:  same as 2nd data file.

        This trains the embedding space in the following manner:

            Let's imagine the 2nd_file_col1 and 3rd_file_col2 are the linkage to col1 in the base
            data set. The training space will effectively be a sparse matrix:

        .. code-block:: text
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

        Parameters
        ----------
        metadata: dict
            Your own dictionary of metadata if you want to track version information or other characteristics.

        ignore_cols:
            A list of columns you want to ignore. You can get the list of columns, which may include renamed columns
            from merging multiple files, by calling detect_only=True

        detect_only:
            If set to True, Featrix will not create the embedding space, but instead it will construct the mapping links,
            detect data types and encoders, identify opportunities for enrichment, and return all of this information
            to you.

        n_epochs: int
            Number of epochs to train on. 5 is the default to let you try things rapidly, but usually at least 25
            epochs and in some cases as many as 5000 may be needed for a high quality embedding space. You can visualize
            the impact of `n_epochs` and `learning_rate` with EZ_PlotLoss().

        learning_rate: float
            Learning rate. Default is 0.01. 0.001 can be useful in some situations.

        capture_training_debug: bool
            Pass in true to capture debug dumps every epoch.

        encoders_override: dict
            Override encoders for specific columns.

        Returns
        -------
        `FeatrixEmbeddingSpace` object representing the new embedding space. Will raise an assertion if an error.
        """
        vector_space_id = self.client.EZ_DataSpaceNewEmbeddingSpace(data_space_id=self.data_space_name,
                                                                    metadata=metadata,
                                                                    ignore_cols=ignore_cols,
                                                                    detect_only=detect_only,
                                                                    n_epochs=n_epochs,
                                                                    learning_rate=learning_rate,
                                                                    print_debug=print_debug)
        if vector_space_id is None:
            return None

        return FeatrixEmbeddingSpace(client=self.client,
                                     vector_space_id=vector_space_id)

class FeatrixEmbeddingSpace:
    """
    FeatrixEmbeddingSpace provides access to all the facilities in an embedding space.
    """
    def __init__(self, client:Featrix, vector_space_id=None):
        assert client is not None and isinstance(client, Featrix)
        if vector_space_id is not None:
            assert type(vector_space_id) == str
        self.vector_space_id = vector_space_id
        self.client = client

    def remove(self, force=False):
        """
        Removes an embedding space, if it exists. No error if it does not exist.

        Does not reset the id on this object.

        Parameters
        ----------
        force: bool
            If set to True, will kill a training process if the embedding space is currently training.
        """
        assert self.vector_space_id is not None, "no vector space id on this object"
        self.client.EZ_RemoveVectorSpace(self.vector_space_id,
                                         force=force)
        return

    def detectEncoders(self,
                       df: pd.DataFrame = None,
                       csv_path: str = None,
                       on_bad_lines: str = 'skip',
                       print_result: bool = True,
                       print_and_return: bool = False
                       ):
        """
        Query Featrix with some data to see how Featrix will interpret the data for encoding
        into the vector embeddings. You can override these detections if needed.

        This will also return information about enriched columns that Featrix will extract.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with both the input and target values.
            This can be the same as the dataframe used to train models in embedding space.

        csv_path: str
            Path to a CSV file to open and read.

        on_bad_lines:
            For reading the CSV file. 'skip', 'error', 'warn'.

        print_result:
            If True, prints the result to the console as a nice table.
            Same as calling `EZ_PrintDetectOnly(EZ_DetectEncoders(...)`.

        print_and_return:
            If True, will print to the console AND return the dictionary.
            Default is False (which means will not return anything without setting print_result=False)

        Returns
        -------
        A dictionary of the printed table; useful for storing or comparing results across
        data sets or different runs. You can ignore this if you are using `print_result=True`.
        """
        return self.client.EZ_DetectEncoders(df=df,
                                             csv_path=csv_path,
                                             on_bad_lines=on_bad_lines,
                                             print_result=print_result,
                                             print_and_return=print_and_return)

    def train(self,
              df: pd.DataFrame = None,
              csv_path: str = None,
              on_bad_lines: str = 'skip',
              ignore_cols=None,
              n_epochs=None,
              learning_rate=None
              ):
        """
        Train a new embedding space on a dataframe. The dataframe should include
        all target columns that you might want to predict.

        You do not need to clean nulls or make the data numeric; pass in strings or missing values
        all that you need to.

        You can pass in timestamps as a string in a variety of formats (ctime, ISO 8601, etc) and
        Featrix will detect and extract features as appropriate.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with both the input and target values.
            This can be the same as the dataframe used to train models in embedding space.

        csv_path: str
            Path to a CSV file to open and read.

        on_bad_lines:
            For reading the CSV file. 'skip', 'error', 'warn'.

        ignore_cols:
            List of columns to ignore when training. If a column is specified that is not found in
            the dataframe, an exception is raised.

        n_epochs: int or None
            Number of epochs to train on. Eventually this will support 'auto'.

        learning_rate: float or None
            Learning rate. Eventually this will support 'auto'.

        sample_percentage: float, 0.0 to 0.1
            How much of the dataframe to load. This will be passed to DataFrame.sample() on the data.

        drop_duplicates: bool
            By default, drops duplicate rows. Set to False if you want to super sample data.

        capture_training_debug: bool
            If set to true, this will capture a dump of the embedding space on every epoch. This is
            useful to create animated visualizations of the convergence of the embedding space.

        encoders_override: dict
            Override auto-detection of encoders.

        Returns
        -------
        str uuid of the embedding space (which we call vector_space_id on other calls)

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:

        .. code-block:: python
            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values

            featrix = Featrix("http://embedding.featrix.com:8080/")
            embeddingSpace = FeatrixEmbeddingSpace(client=featrix)
            embeddingSpace.train(all_df, ...)
        """
        self.vector_space_id = self.client.EZ_NewEmbeddingSpace(df=df,
                                                                csv_path=csv_path,
                                                                vector_space_id=self.vector_space_id,
                                                                on_bad_lines=on_bad_lines,
                                                                ignore_cols=ignore_cols,
                                                                n_epochs=n_epochs,
                                                                learning_rate=learning_rate)
        return self.vector_space_id

    def continueTraining(self,
                         n_epochs: int,
                         learning_rate: float = 0.01):
        """
        Continue training the embedding space.

        You can query the training history and loss by checking the structure returned from `EZ_VectorSpaceMetaData()`.

        Parameters
        ----------
        epochs: int
            Number of epochs.

        learning_rate: float
            Learning rate. 0.01 is the default.

        capture_training_debug: bool
            Pass in true to capture debug dumps every epoch.
        """
        self.client.EZ_EmbeddingSpaceContinueTraining(self.vector_space_id,
                                                   n_epochs=n_epochs,
                                                   learning_rate=learning_rate)
        return

    def metadata(self):
        """
        Get metadata for the specified embedding space.

        Parameters
        ----------
        None

        Returns
        -------
        A dictionary of metadata. The dictionary contains:
            Information about the columns used to train the embedding space.
            The training time, batch dimensions, and other statistics.
            Which encoders were used for which columns and the detected probability of data types.
            Information about every set of training done on the embedding space, including loss per iteration and number of epochs.

        The specific format of the dictionary may change from release to release as we improve the product.

        Returns None if the embedding space does not exist.
        """
        return self.client.EZ_EmbeddingSpaceMetaData(self.vector_space_id)

    def columns(self):
        """
        Retrieve the list of columns that were embedded in the embedding space. The embedding space
        must have already been trained using the `train()` method.

        If Featrix was unable to process a column, then it will not be in the list.

        Parameters
        ----------
        None

        Returns
        -------
        A list of column names
        """
        return self.client.EZ_EmbeddingSpaceColumns(self.vector_space_id)

    def plotLoss(self):
        """
        Show a matplotlib plot of the loss on training the embedding space.
        """
        return self.client.EZ_PlotLoss(self.vector_space_id)

    def plotEmbeddings(self,
                       col1_name,
                       col2_name=None,
                       col1_range=None,
                       col2_range=None,
                       col1_steps=None,
                       col2_steps=None,
                       relative_scale=False,
                       axis_label_precision=5,
                       show_unknown=False):
        """
        Plot similarity plots of embeddings in the embedding space.

        Parameters
        ----------
        col1_name: str
            The first field in the embedding space to plot.

        col2_name: str
            This can be specified to show the similarity of embeddings between two columns.
            If this is NOT specified, then this will produce a self-similarity plot of col1_name vs col1_name itself.

        col1_range: (min, max) tuple
            Range of values, used if col1 is a scalar. Default is mean ± 2 * std.

        col2_range: (min, max) tuple
            Range of values, used if col2 is a scalar. Default is mean ± 2 * std.

        col1_steps: int
            Number of steps to sample across col1_range, if col1 is a scalar.

        col2_steps: int
            Number of steps to sample across col2_range, if col2 is a scalar.

        relative_scale: bool
            If true, a relative scale will be used. Defaults to False; the default scale is [-1, 1].

        rotate_x_labels: bool
            If true, will turn the X labels to be easier to read.

        axis_label_precision: int
            The number of decimal places to show the scalar axis labels. Set to 0 to round to integers.

        show_unknown: bool
            Show the <UNKNOWN> token for sets. The unknown is a special value used when embedding sets, which provides
            a lot of power to Featrix, but it can make for some confusing visualizations.

        show_metadata: bool
            If true, will show the time the embedding space was trained.

        epoch_index: int
            Show a specific view at the end of the specified epoch during training.

        animate_training: bool
            If true, will create an animated gif showing the training of the vectors.

        Raises
        ------
        FeatrixEmbeddingSpaceNotFound
            If the embedding space specified doesn't exist.

        FeatrixColumnNotFound
            If a specified column name doesn't exist in the embedding space.

        FeatrixColumnNotAvailable
            If a specified column is not available for distance plotting.
        """
        return self.client.EZ_PlotEmbeddings(self.vector_space_id,
                                             col1_name=col1_name,
                                             col2_name=col2_name,
                                             col1_range=col1_range,
                                             col2_range=col2_range,
                                             col1_steps=col1_steps,
                                             col2_steps=col2_steps,
                                             relative_scale=relative_scale,
                                             axis_label_precision=axis_label_precision,
                                             show_unknown=show_unknown)

    def cluster(self,
                k:int,
                columns: [str] = None,
                forceNewCluster:bool=False,
                return_centroids=False,
                return_columns: [str] = None,
                n_epochs:int=10_000
                ):
        """
        Cluster the training data.

        Future versions of the API will let you cluster new data sets, with and without the training data.

        This call will block until the clustering is complete; this may take a few seconds or minutes.

        Parameters
        ----------
        k: int
            The number of clusters. There is an art to picking k and we will add tools later
            to help. If you pick k to be too small, many clusters will appear to be similar.

            Note that this implementation does *not* split the data into equal sized k groups.

        columns:
            The column names to cluster on.

            NOTE: The current release supports just one column. This will be fixed ASAP.

            This defaults to None. When None is used, the cluster index will be built considering
            all columns. Often times we want to experiment with different arrangements and this
            lets us do that--we can verify clusters on a single column or a reduced subset
            without training a new embedding space.

        return_centroids:
            Return the centroid coordinates of the clusters. This can be used to evaluate the cluster
            separation.

        return_columns:
            List of columns to return in the returned data. This lets you get back just the columns
            you want to evaluate the clusters. If None, then all data fields will be returned.

        n_epochs:
            Number of epochs to train the clustering on.

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
        return self.client.EZ_EmbeddingSpaceCluster(self.vector_space_id,
                                                    k=k,
                                                    columns=columns,
                                                    forceNewCluster=forceNewCluster,
                                                    return_centroids=return_centroids,
                                                    return_columns=return_columns,
                                                    n_epochs=n_epochs)

    def embed(self, *args, **kwargs):
        return self.embedRecords(*args, **kwargs)

    def embedRecords(self,
                     records = None,
                     colList: [list] = None):
        """
        Embed new records. You can use this to test embeddings for new data that isn't trained in the embedding space
        (or that is); you can pass partial records, the sky is the limit.

        This does not edit the embedding space.

        Parameters
        ----------
        records:
            This can be a dataframe or a list of dictionaries.

            The keys in the dictionary need to be column names from the embedding space (which you can query with
            `columns()`.

        colList:
            A list of keys. You can use this to pass only some of the fields in the records argument without
            having to manually drop or reduce the data.
        """
        return self.client.EZ_EmbedRecords(vector_space_id=self.vector_space_id,
                                           records=records,
                                           colList=colList)

    def newModel(self):
        """
        Creates a new model object using this embedding space and the Featrix client object in use by this embedding space.

        Call train() on the returned new model to do useful things with it and instantiate it on the server.
        """
        return FeatrixModel(embeddingSpace=self)

class FeatrixModel:
    def __init__(self,
                 embeddingSpace=None,
                 client=None,
                 vector_space_id=None,
                 model_id=None):
        self.vector_space_id = vector_space_id
        self.client = client
        if embeddingSpace is not None:
            assert isinstance(embeddingSpace, FeatrixEmbeddingSpace)
            assert vector_space_id is None, "cannot specify both vector_space_id AND embeddingSpace"
            assert client is None, "cannot specify both client AND embeddingSpace"
            self.vector_space_id = embeddingSpace.vector_space_id
            self.client = embeddingSpace.client
        else:
            assert vector_space_id is not None, "must specify vector_space_id if not using embeddingSpace object"
            assert client is not None, "must specify client if not using embeddingSpace object"
            self.vector_space_id = vector_space_id
            self.model_id = model_id

    def train(self,
              target_column_name: str,
              df: pd.DataFrame | list[pd.DataFrame],
              n_epochs: int = 25,
              size: str = 'small'):
        """
        Create a new model in a given embedding space.

        Parameters
        ----------
        target_column_name : str
            Name of the target column. Must be present in the passed DataFrame `df` or the passed DataFrame dictionary after mapping.

        df : pd.DataFrame | list[pd.DataFrame]
            The dataframe with both the input and target values. This can be the same as the dataframe
            used to train the embedding space.

            For embedding spaces created from joined data in data spaces, this can be a list of data frames
            to train the model.

        n_epochs: int
            Number of epochs to train on.

        size: str
            Can be 'small', 'large'
            For models that run in the Featrix server, 'small' is a 2 hidden layer model with 50 dimensions.
            'large' is a 6 layer model with 50 dimensions.

        Returns
        -------
        str uuid of the model (which we call model_id on other calls)

        Raises
        ------
        FeatrixEmbeddingSpaceNotFound
            if the embedding space specified does not exist.

        Notes
        -----
        This call blocks until training has completed; the lower level API gives you more async control.

        To combine a series and dataframe, do something like:

            all_df = df.copy() # call .copy() if you do not want to change the original.
            all_df[target_col] = target_values
        """
        self.model_id = self.client.EZ_NewModel(vector_space_id=self.vector_space_id,
                                                target_column_name=target_column_name,
                                                df=df,
                                                n_epochs=n_epochs,
                                                size=size)

    def checkGuardrails(self, query, issues_only=False):
        """
        Checks the parameters of the query for potential inconsistencies between
        the query and what the embedding space has been trained on.

        Warnings or errors from this do not mean running a prediction will fail,
        but they can indicate that the query is beyond the bounds of what
        has been trained and therefore results may be unexpected.

        Use this for debugging and getting a feel for the embedding space shapes.

        This call is designed to be interchangeable with `predict()`.

        Parameters
        ----------
        query : dict or [dict]
            This is exactly what you would pass to `predict()`
            Either a single parameter or a list of parameters.
            { col1: <value> }, { col2: <value> }

        issues_only: bool
            If True, will return only warnings and errors and no informative messages.
        """
        return self.client.EZ_ModelCheckGuardrails(vector_space_id=self.vector_space_id,
                                                   model_id=self.model_id,
                                                   query=query,
                                                   issues_only=issues_only)

    def predict(self, query):
        """
        Predict a probability distribution on a given model in a embedding space.

        Query can be a list of dictionaries or a dictionary for a single query.

        Parameters
        ----------
        query : dict or [dict]
            Either a single parameter or a list of parameters.
            { col1: <value> }, { col2: <value> }

        check_guardrails: bool
            If True, will run `checkGuardrails()` first, and print out any errors
            or warnings to the console.

        Returns
        -------
        A dictionary of values of the model's target_column and the probability of each of those values occurring.
        """
        return self.client.EZ_Prediction(vector_space_id=self.vector_space_id,
                                         model_id=self.model_id,
                                         query=query)

    def predictOnDataFrame(self,
                           df: pd.DataFrame,
                           target_column: str = None,
                           include_probabilities: bool = False,
                           check_accuracy: bool = False,
                           print_info: bool = False,
                           ):
        """
        Given a dataframe, treat the rows as queries and run the given model to provide a prediction on the target
        specified when creating the model.

        Parameters
        ----------
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

        Notes
        -----
        In this version of the API, queries are for categorical values only.
        """
        return self.client.EZ_PredictionOnDataFrame(vector_space_id=self.vector_space_id,
                                                    model_id=self.model_id,
                                                    df=df,
                                                    target_column=target_column,
                                                    include_probabilities=include_probabilities,
                                                    check_accuracy=check_accuracy,
                                                    print_info=print_info)



