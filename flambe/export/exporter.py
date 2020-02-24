import tempfile
import os
import boto3
import subprocess
from urllib.parse import urlparse
from typing import Any

import flambe
from flambe.compile import Schema, Registrable, YAMLLoadType
from flambe.const import DEFAULT_PROTOCOL
from flambe.logging import coloredlogs as cl

import logging

logger = logging.getLogger(__name__)


class Exporter(Registrable):
    """Implement a Exporter.

    An Exporter is a simple object that can be used to create
    any Component post-experiment, and export it to a local
    or remote location.

    Currently supports local, and S3 locations.

    """
    def __init__(self,
                 obj: Schema,
                 override: bool = False,
                 storage: str = 'local',
                 compress: bool = False,
                 pickle_only: bool = False,
                 pickle_protocol=DEFAULT_PROTOCOL) -> None:
        """Initialize the Exporter.

        Parameters
        ----------
        obj : Schema
            The object to build, and export
        destination : str
            The destination where the object should be saved.
            If an s3 bucket is specified, 's3' should also be
            specified as the storage argument. s3 destinations
            should have the following syntax:
            's3://<bucket-name>[/path/to/folder]'
        storage: str
            The storage location. One of: [local | s3]
        compress : bool
            Whether to compress the save file / directory via tar + gz
        pickle_only : bool
            Use given pickle_module instead of the hiearchical save
            format (the default is False).
        pickle_module : type
            Pickle module that has load and dump methods; dump should
            accept a pickle_protocol parameter (the default is dill).
        pickle_protocol : type
            Pickle protocol to use; see pickle for more details (the
            default is 2).

        """
        super().__init__()

        self.override = override
        self.obj = obj

        self.compiled_obj: Any

        self.storage = storage
        # self.serialization_args = {
        #     'compress': compress,
        #     'pickle_only': pickle_only,
        #     'pickle_module': pickle_module,
        #     'pickle_protocol': pickle_protocol
        # }

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS

    def run(self) -> bool:
        """Run the Exporter."""
        env = flambe.get_env()
        # Add information about the extensions. This ensures
        # the compiled obj has the extensions information
        # self.obj.add_extensions_metadata(self.extensions)
        self.compiled_obj = self.obj()  # Compile Schema

        if self.storage == 'local':
            self.save_local(env.output_path)
        elif self.storage == 's3':
            self.save_s3(env.output_path)
        else:
            msg = f"Unknown storage {self.storage}, should be one of: [local, s3]"
            raise ValueError(msg)

        return False

    def save_local(self, path) -> None:
        """Save an object locally.

        Parameters
        ----------
        force: bool
            Wheter to use a non-empty folder or not

        """
        force = self.override
        if (
            os.path.exists(path) and
            os.listdir(path) and
            not force
        ):
            raise ValueError(
                f"Destination {path} folder is not empty. " +
                "Use --force to force the usage of this folder or " +
                "pick another destination."
            )

        # TODO: switch flambe.save
        out_path = os.path.join(path, 'checkpoint.pt')
        flambe.save(self.compiled_obj, out_path)

    def get_boto_session(self):
        """Get a boto Session

        """
        return boto3.Session()

    def save_s3(self, path) -> None:
        """Save an object to s3 using awscli

        Parameters
        ----------
        force: bool
            Wheter to use a non-empty bucket folder or not

        """
        force = self.override

        url = urlparse(path)

        if url.scheme != 's3' or url.netloc == '':
            raise ValueError(
                "When uploading to s3, destination should be: " +
                "s3://<bucket-name>[/path/to/dir]"
            )

        bucket_name = url.netloc
        s3 = self.get_boto_session().resource('s3')
        bucket = s3.Bucket(bucket_name)

        for content in bucket.objects.all():
            path = url.path[1:]  # Remove first '/'
            if content.key.startswith(path) and not force:
                raise ValueError(
                    f"Destination {path} is not empty. " +
                    "Use --force to force the usage of this bucket folder or " +
                    "pick another destination."
                )

        with tempfile.TemporaryDirectory() as tmpdirname:
            flambe.save(self.compiled_obj, tmpdirname)
            # TODO fix don't use flambe save; also probably have one helper for the save operation
            # so that local and remote do the exact same thing
            # flambe.save(self.compiled_obj, tmpdirname, **self.serialization_args)
            try:
                subprocess.check_output(
                    f"aws s3 cp --recursive {tmpdirname} {path}".split(),
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            except subprocess.CalledProcessError as exc:
                logger.debug(exc.output)
                raise ValueError(f"Error uploading artifacts to s3. " +
                                 "Check logs for more information")
            else:
                logger.info(cl.BL(f"Done uploading to {path}"))
