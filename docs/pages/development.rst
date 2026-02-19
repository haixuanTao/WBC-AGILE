Development
===========

This page covers the development workflow including code style, pre-commit hooks, testing,
and contribution guidelines.


Code Style and Standards
------------------------

- **Python:** `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ with modifications from
  the project's Ruff configuration (120-character line length).
- **Formatting:** Ruff formatter (Black-compatible) and isort for import sorting.
- **Type Hints:** Encouraged for readability and maintainability.


Pre-commit Hooks
----------------

This repository uses pre-commit hooks to ensure code quality.

**Setup:**

.. code-block:: bash

   ./scripts/setup/setup_hooks.sh

**Run manually:**

.. code-block:: bash

   pre-commit run --all-files

The pre-commit configuration includes:

- Code formatting with Ruff (Black-compatible) and isort
- Linting with Flake8
- Type checking with mypy
- File checks (trailing whitespace, merge conflicts, etc.)

The ``third_party`` directory is excluded from all hooks.


Testing
-------

Test Types
^^^^^^^^^^

**Unit Tests** (in ``agile/rl_env/tests/``):

- Test individual MDP components (actions, rewards, terminations, etc.).
- Run automatically in CI on every push.
- Quick to execute (~1 minute).

**End-to-End (E2E) Tests** (in ``tests/``):

- ``test_all_tasks_e2e.py``: Complete training pipelines for all tasks.
- ``test_deterministic_eval_e2e.py``: Deterministic evaluation pipeline.
- Run on main branch or manually triggered.


Running Tests
^^^^^^^^^^^^^

**Docker Testing (Recommended — matches CI):**

.. code-block:: bash

   # Run ALL tests (unit + E2E)
   ./tests/test_e2e_ci_locally.sh --all

   # Run only E2E tests
   ./tests/test_e2e_ci_locally.sh

   # Run only unit tests
   ./tests/test_e2e_ci_locally.sh --unit

   # Test a specific task
   ./tests/test_e2e_ci_locally.sh --task Velocity-G1-v0

**Local Testing (requires Isaac Lab):**

.. code-block:: bash

   # Unit tests
   ./tests/run_unit_tests.sh

   # E2E tests (requires GPU)
   ${ISAACLAB_PATH}/isaaclab.sh -p tests/test_all_tasks_e2e.py

   # Deterministic evaluation E2E test
   ${ISAACLAB_PATH}/isaaclab.sh -p tests/test_deterministic_eval_e2e.py


Adding Tests for New Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new task, add it to the E2E test suite:

1. Register the task in ``agile/rl_env/tasks/<category>/<robot>/__init__.py``.

2. Add to E2E tests in ``tests/test_all_tasks_e2e.py``:

   .. code-block:: python

      # Find the section marked with "ADD YOUR NEW TASKS HERE"
      "YourTask-Robot-v0",  # Brief description

3. Test locally before pushing:

   .. code-block:: bash

      ./tests/test_e2e_ci_locally.sh --task YourTask-Robot-v0


CI Pipeline
^^^^^^^^^^^

The CI pipeline runs in three stages:

1. **Lint** — Code quality checks (always runs).
2. **Unit Tests** — Component testing (always runs).
3. **E2E Tests** — Full training tests (main branch or manual trigger, 30-minute timeout).


Contributing
------------

1. Fork the repository and create a new branch from ``main``.
2. Make changes following the code style guidelines.
3. Add tests for your changes if applicable.
4. Update documentation as necessary.
5. Run pre-commit hooks:

   .. code-block:: bash

      pre-commit run --all-files

6. Submit a pull request with a clear description.

**Sign your commits** using the ``--signoff`` (``-s``) flag:

.. code-block:: bash

   git commit -s -m "Add cool feature."

All contributions require sign-off under the Developer Certificate of Origin (DCO).
See :agile_code_link:`<CONTRIBUTING.md>` for full details.
