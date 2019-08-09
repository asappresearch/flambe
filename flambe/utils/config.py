import os
import re
from typing import Dict

import jinja2


def generate_config_from_template(template_path: str,
                                  config_path: str,
                                  remove_comments: bool = False,
                                  **template_kwargs: Dict[str, str]):
    """
    Parameters
    ----------
    template_path: str
        The path to the config template
    config_path: str
        The path to which the rendered config should be written
    remove_comments: bool
        If `True`, removes comments from the rendered config before
        writing it to disk
    template_kwargs:
        Keyword arguments to pass to your template, e.g.
        `path='config.yaml', foo='bar'`

    Example config:

    ```yaml
    !Experiment

    foo: {{ bar }}
        baz: {{ skittles }}
    ```

    If saved as config.yaml.template, then invoking:

    ```python
    generate_config_from_template('config.yaml.template',
        'config.yaml', bar='pickles', skittles='yum')
    ```

    the following config will be written to 'config.yaml':

    ```yaml
    !Experiment

    foo: pickles
        baz: yum
    ```
    """
    dirname = os.path.dirname(template_path)
    basename = os.path.basename(template_path)
    loader = jinja2.FileSystemLoader(searchpath=dirname)
    env = jinja2.Environment(loader=loader, autoescape=True)
    template = env.get_template(basename)
    with open(config_path, 'w') as f:
        for line in template.render(**template_kwargs).split('\n'):
            if remove_comments:
                line = re.sub('# .*', '', line).rstrip()
            if line:
                f.write(line + '\n')
