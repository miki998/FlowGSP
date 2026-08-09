Contributing
============

Contributions are welcome — bug reports, new operators and filters, examples,
and documentation all help.

Before you start, read `CONTRIBUTING.md
<https://github.com/miki998/GyRAPH/blob/main/CONTRIBUTING.md>`_ for the branch
and commit conventions and the review process, and `CODE_OF_CONDUCT.md
<https://github.com/miki998/GyRAPH/blob/main/CODE_OF_CONDUCT.md>`_.

Setting up
----------

.. code-block:: bash

   git clone https://github.com/miki998/GyRAPH.git
   cd GyRAPH
   pip install -e .
   pip install pytest pytest-cov flake8 pre-commit
   pre-commit install

Checks that CI runs
-------------------

.. code-block:: bash

   coverage run -m unittest discover -s tests/ -p 'test_*.py'
   coverage report -m
   flake8 . --max-line-length=127

Type checking is configured in `mypy.ini
<https://github.com/miki998/GyRAPH/blob/main/mypy.ini>`_.

Documentation
-------------

Docstrings are **NumPy style** — that is what the API reference is generated
from, so a new public function is not finished until it has one:

.. code-block:: python

   def dirichlet(signal, L, normalize=True):
       """
       Compute the Dirichlet energy of a signal with respect to a graph Laplacian.

       Parameters
       ----------
       signal : np.ndarray, shape (N,)
           Signal defined on the graph vertices.
       L : np.ndarray, shape (N, N)
           Graph Laplacian.
       normalize : bool, optional
           Whether to normalise by the signal energy, by default True.

       Returns
       -------
       float
           The Dirichlet energy.
       """

Build the docs locally before opening a pull request:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

When you add a new public class or function, add it to the matching page under
``docs/reference/``. The site rebuilds automatically on every push to ``main``.
