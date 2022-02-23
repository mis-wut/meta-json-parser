# Notebooks

This directory holds various notebooks that are used to show and
analyse benchmark results, and results of various tests.

## Contributing

[Jupyter Notebooks][jupyter] in this directory use **[nbdev][]** library
to make it easier to work with notebooks in a version-controlled
manner.  You don't need this library to view notebooks.

Actually, the nbdev library is not here to have to use Jupyter as
development environment to create Python packages (nbdev intended use)
but for its git hooks, to reduce the chance for a merge conflict,
and to avoid adding spurious changes to the repository.

[jupyter]: https://jupyter.org/ "Project Jupyter | interactive data science and scientific computing"
[nbdev]: https://nbdev.fast.ai/ "Welcome to nbdev | Create delightful Python projects using Jupyter Notebooks"

### Installing nbdev

nbdev is on [PyPI][] and [conda][] so you can just run
`pip install nbdev` (or `python -m pip install nbdev`)
or `conda install -c fastai nbdev`.
Microsoft Windows users should use `pip` instead of `conda`
to install nbdev.

To make sure that nbdev is installed in the same environment
as Jupyter, it is recommended to do this from the terminal window
in Jupyter Notebook or JupyterLab.

[PyPI]: https://pypi.org/ "PyPI · The Python Package Index"
[conda]: https://docs.conda.io/ "Conda | Package, dependency and environment management for any language"

### Install git hooks to avoid and handle conflicts

Jupyter Notebooks can cause challenges with Git conflicts, but you
can mitigate the issue with the help of [filter drivers][filter].
The first step is to set up "hooks" which will remove metadata from
your notebooks when you commit, greatly reducing the chance
you have a conflict. This also helps in examining the actual changes
to the contents of the notebook, as opposed to the changes caused
by running it.

This is done with the help of _clean-nbs_ filter.  The `.gitattributes`
file declares it as a clean / smudge filter for commit / checkout
operation, respectively,  for all `*.ipynb` files
with the following line:
```
**/*.ipynb filter=clean-nbs
```

The other part is defining the filter, which is done in the
`.gitconfig_nbdev` file.  The relevant parts are:
```
[filter "clean-nbs"]
        clean = nbdev_clean_nbs --read_input_stream True
        smudge = cat
        required = true
```
The `nbdev_clean_nbs` command comes from the nbdev library.

To activate the filter you need to [include][gitconfig-include] it
in one of Git configuration files.  This can be done with the
following command:
```
$ git config --local include.path ../benchmark/notebooks/.gitconfig_nbdev
```

If you get a _**merge conflict**_, with nbdev you can simply run
`nbdev_fix_merge examine_benchmarks.ipynb`. This will replace any
conflicts in cell outputs with your version, and if there are
conflicts in input cells, then both cells will be included in the
merged file, along with standard conflict markers (e.g. `=====`).
Then you can open the notebook in Jupyter and choose which version
to keep.

To make it easier to view differences in Jupyter notebooks,
the `.gitattributes` file defines `diff` attribute for the
`*.ipynb` files:
```
**/*.ipynb diff=ipynb
```
The `ipynb` diff driver is defined also in `.gitconfig_nbdev`
in the following way:
```
[diff "ipynb"]
        textconv = nbdev_clean_nbs --disp True --fname
```
This defines how to do the diff of `*.ipynb` files: use
the [textconv][] "hook" to remove the output (clean them), and then
show the differences as if they were ordinary text file(s).

[filter]: https://git-scm.com/docs/gitattributes#_filter "gitattributes - Defining attributes per path # filter attribute"
[textconv]: https://git-scm.com/docs/gitattributes#_performing_text_diffs_of_binary_files "gitattributes - Defining attributes per path # Performing text diffs of binary files"
[gitconfig-include]: https://www.git-scm.com/docs/git-config#_includes "git-config - Get and set repository or global options # Includes"