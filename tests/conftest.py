import os
import pytest
import convert
import shutil

from . import IMG_PATH
from . import ARR_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = ''


@pytest.fixture(scope="session")
def conversion():
    args = convert.parser().parse_args(['-i', IMG_PATH,
                                        '-o', ARR_PATH, 'numpy'])

    # Convert images to numpy arrays
    convert.main(args)

    yield None

    # Cleanup generated numpy arrays
    shutil.rmtree(ARR_PATH)


