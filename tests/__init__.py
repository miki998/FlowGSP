import unittest
import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"src"
)
sys.path.append(SOURCE_PATH)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test*.py')

    runner = unittest.TextTestRunner()
    runner.run(suite)