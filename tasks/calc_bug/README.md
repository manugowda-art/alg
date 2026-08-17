# task: calc_bug

A four-function statistics module with a failing test suite. Two defects are
seeded in `calc/stats.py`; one of them makes two tests fail, so the failure
count does not map one-to-one onto the number of bugs.

The tests are the specification. A run counts as solved when `pytest` is green
and no test file has been modified.
