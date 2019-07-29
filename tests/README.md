The [pytest](https://docs.pytest.org/en/latest/contents.html#toc) framework will be used for this project.
This document will highlight the main concepts to keep in mind. For details, more nuanced cases, or examples, click through the `[Read more]` sections.


##### Key points
* All files should be named test\_\*.py or \*\_test.py. All test functions should also be prefaced with test\_\*. \[[Read more](https://docs.pytest.org/en/latest/goodpractices.html#test-discovery)\]
* Use fixtures for shared functions and objects. \[[Read more](https://docs.pytest.org/en/latest/fixture.html#fixture)\]
* Parameterize functions to avoid writing duplciate code. \[[Read more](https://docs.pytest.org/en/latest/parametrize.html)\]
* Use attributes to mark special test cases. \[[Read more](https://docs.pytest.org/en/latest/mark.html#mark)\]
* This is the dedicated `test` directory. It will exactly mimic the layout of the main directory. Tests for any specific module will be placed within its corresponding folder/sub-folders within this directory. \[[Read more](https://docs.pytest.org/en/latest/goodpractices.html#tests-outside-application-code)\]
  <br>For example:
  ```
  module1/
  module2/
  file1.py
  tests/
    module1/
    module2/
    test_file1.py
  ```
* Any data files for tests will live in the `data` directory.
* There should be one concept per test. This might take the form of multiple assertions, but only one main thing should be tested. It needs to be clear from the tests what exactly failed.

##### Running

Simply call `pytest [file/directory]` via command line. \[[Read more](https://docs.pytest.org/en/latest/usage.html)\]

##### Plugins
* [pytest-pep8](https://pypi.org/project/pytest-pep8/): check pep8 compliance for all files
* [pytest-cov](https://pypi.org/project/pytest-cov/): automatically produce coverage reports
* [pytest-mock](https://github.com/pytest-dev/pytest-mock): monkeypatch fixtures
