from typing import Optional
import os
import subprocess
import socket

import torch

from flambe.search.protocol import Searchable


class Checkpoint(object):

    def __init__(self,
                 path: str,
                 host: Optional[str] = None,
                 user: Optional[str] = None):
        """Initialize a checkpoint.

        Parameters
        ----------
        path : str
            The local path used for saving
        host : str, optional
            An optional host to upload the checkpoint to,
            by default None
        user: str, optional
            An optional user to use alongside the host name,
            by default None

        """
        self.path = path
        self.host = host
        self.checkpoint_path = os.path.join(self.path, 'checkpoint.pt')
        self.remote = f"{user}@{host}:{self.checkpoint_path}" if host else None

    def get(self) -> Searchable:
        """Retrieve the object from a checkpoint.

        Returns
        -------
        Searchable
            The restored Searchable object.

        """
        if os.path.exists(self.checkpoint_path):
            searchable = torch.load(self.checkpoint_path)
        else:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            if self.remote:
                subprocess.run(f'rsync -az -e "ssh -i $HOME/ray_bootstrap_key.pem" \
                    {self.remote} {self.checkpoint_path}')
                searchable = torch.load(self.checkpoint_path)
            else:
                raise ValueError(f"Checkpoint {self.checkpoint_path} couldn't be found.")
        return searchable

    def set(self, searchable: Searchable):
        """Retrieve the object from a checkpoint.

        Parameters
        ----------
        Searchable
            The Searchable object to save.

        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(searchable, self.checkpoint_path)
        if self.remote:
            current_ip = socket.gethostbyname(socket.gethostname())
            if str(current_ip) != self.host:
                subprocess.run(f'rsync -az -e "ssh -i $HOME/ray_bootstrap_key.pem" \
                    {self.checkpoint_path} {self.remote}')
