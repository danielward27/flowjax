### Documentation

The documentation is supported by [Sphinx](https://www.sphinx-doc.org/en/master/). To build the HTML pages locally, run

```
make -C docs html
```
from the docs directory. The documentation can then be viewed by opening `./docs/_build/html/index.html``. To test the doctest code blocks in the documentation, run from the top level directory:
```
make -C docs doctest
```

Github Actions is used for continuous integration, and the tests will fail if either the documentation does not build, or any doctest examples fail.
