
### Documentation

The documentation is supported by [Sphinx](https://www.sphinx-doc.org/en/master/). To build the docs, run from the top level directory:
```
sphinx-build docs docs/_build
```

To test the doctest code blocks in the documentation, run from the top level directory:
```
make -C docs doctest
```

Github Actions is used for continuous integration, and the tests will fail if either the documentation does not build, or any doctest examples fail.
