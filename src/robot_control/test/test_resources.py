
import pytest
from glob import glob
import os

def test_resources():

    assert "a" in (os.listdir("/home/cotrobot/resources/")), 'a folder not found'

if __name__=="__main__":
    test_resources()