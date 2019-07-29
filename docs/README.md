## How to compile documentation

#### Install requirements

```bash
$ pip install -r docs/requirements.txt
```

Run:

```bash
$ make html
```

This will generate a `_build` folder with the `html` files in it


#### Rebuilding RST's

```bash
$ sphinx-apidoc -fMeT -o source ../flambe
```

Then:

```bash
$ rm source/flambe.rst
$ rm source/flambe.logo.rst
$ rm source/flambe.version.rst
```
