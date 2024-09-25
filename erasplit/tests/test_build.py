import os
import pytest
import textwrap

from erasplit import __version__
from erasplit.utils._openmp_helpers import _openmp_parallelism_enabled


def test_openmp_parallelism_enabled():
    # Check that erasplit is built with OpenMP-based parallelism enabled.
    # This test can be skipped by setting the environment variable
    # ``erasplit_SKIP_OPENMP_TEST``.
    if os.getenv("erasplit_SKIP_OPENMP_TEST"):
        pytest.skip("test explicitly skipped (erasplit_SKIP_OPENMP_TEST)")

    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    err_msg = textwrap.dedent(
        """
        This test fails because scikit-learn has been built without OpenMP.
        This is not recommended since some estimators will run in sequential
        mode instead of leveraging thread-based parallelism.

        You can find instructions to build scikit-learn with OpenMP at this
        address:

            https://scikit-learn.org/{}/developers/advanced_installation.html

        You can skip this test by setting the environment variable
        erasplit_SKIP_OPENMP_TEST to any value.
        """
    ).format(base_url)

    assert _openmp_parallelism_enabled(), err_msg
