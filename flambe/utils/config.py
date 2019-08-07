import os
import re

import jinja2


def generate_config_from_template(template_path: str,
                                  config_path: str,
                                  remove_comments: bool = False,
                                  **template_kwargs):
    """
    Example config:

    ```yaml
    !Experiment

    foo: {{ bar }}
        baz: {{ skittles }}
    ```

    If saved as config.yaml.template, then invoking:

    ```python
    generate_config_from_template('config.yaml.template', 'config.yaml', bar='pickles', skittles='yum')
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
    env = jinja2.Environment(loader=loader)
    template = env.get_template(basename)
    with open(config_path, 'w') as f:
        for line in template.render(**template_kwargs).split('\n'):
            if remove_comments:
                line = re.sub('# .*', '', line).rstrip()
            if line:
                f.write(line + '\n')
