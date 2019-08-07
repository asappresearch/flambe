import os
import re

import jinja2


def generate_config_from_template(template_path, config_path, remove_comments=False, **template_kwargs):
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
