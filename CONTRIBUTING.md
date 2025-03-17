# How to contribute to enzax

All contributions are very welcome!

Make sure to read the [code of conduct](https://github.com/dtu-qmcm/CODE_OF_CONDUCT.md) and follow its recommendations.

If you have a specific suggestion for how enzax could be improved, or if you find a bug then please file an issue or submit a pull request.

Alternatively, if you have any more general thoughts or questions, please post them in the [discussions page](https://github.com/dtu-qmcmc/enzax/discussions).

If you would like to contribute code changes, just follow the normal [GitHub workflow](https://docs.github.com/en/get-started/quickstart/github-flow): make a local branch with the changes then create a pull request.

## Developing enzax locally

To develop enzax locally you will probably need to install it with development dependencies. Here is how to do so:

```sh
$ pip install enzax'[dev]'
```

You can see what these dependencies are by checking the `[dependencies]` table in enzax's [`pyproject.toml` file](https://github.com/dtu-qmcm/enzax/blob/main/pyproject.toml).

## Releasing new versions of enzax

To release a new version of enzax, edit the field `version` in `pyproject.toml`, e.g. to `0.2.1` then make a pull request with this change.

Once the changes are merged into the `origin/main` branch, add a tag whose name begins with `v`, followed by the new version number to your local `main` branch, for example like this:

```sh
$ git tag v0.2.1
```

Now push the new tag to GitHub:

```sh
$ git push origin "v0.2.1"
```
