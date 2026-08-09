Installation
============

GyRAPH requires **Python 3.9 or newer**.

From PyPI
---------

.. code-block:: bash

   pip install GyRAPH

From source
-----------

.. code-block:: bash

   git clone https://github.com/miki998/GyRAPH.git
   cd GyRAPH
   pip install -e .

Dependencies
------------

The core of the library — graphs, operators, filters, surrogates and metrics —
runs on NumPy, SciPy, NetworkX, scikit-learn, pandas, SymPy, tqdm and the
matplotlib/seaborn/scienceplots stack. These are installed automatically.

A few optional-in-spirit modules pull in heavier dependencies:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Needed by
   * - ``torch``
     - the gradient-descent coefficient fitting in
       :meth:`~gyraph.filters.PolynomialFilter.regression_descent` and its
       Chebyshev/Faber counterparts
   * - ``opencv-python``
     - :func:`~gyraph.utils.create_video_from_images` for rendering dynamics
   * - ``scikit-image``, ``torch-geometric``, ``torchvision``
     - mesh and learning-oriented helpers

They are declared as install requirements today, so a plain ``pip install``
brings everything.

Checking the installation
-------------------------

.. code-block:: python

   import gyraph
   print(gyraph.__version__)

.. note::

   Importing ``gyraph`` applies a publication-oriented matplotlib style
   (``science`` / ``ieee``) and sets ``figure.dpi`` to 300. If you would rather
   keep your own ``rcParams``, restore them after the import:

   .. code-block:: python

      import matplotlib as mpl
      import gyraph
      mpl.rcdefaults()

   See ``gyraph/constants.py`` for exactly what is set.

Development install
-------------------

.. code-block:: bash

   pip install -e .
   pip install pytest pytest-cov flake8 pre-commit
   pre-commit install

Run the test suite:

.. code-block:: bash

   python -m unittest discover -s tests/ -p 'test_*.py'

With coverage and linting, as CI does:

.. code-block:: bash

   coverage run -m unittest discover -s tests/ -p 'test_*.py'
   coverage report -m
   flake8 . --max-line-length=127

Building the documentation
--------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

Then open ``docs/_build/html/index.html``.
