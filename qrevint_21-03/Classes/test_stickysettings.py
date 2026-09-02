import pytest
import os
import sys
from Classes.stickysettings import StickySettings as ss


def create_filename():
    """Create filename to use in testing"""
    file_exists = True
    n = 0
    testfile = ''
    while file_exists:
        n += 1
        testfile = os.path.join(os.getenv('APPDATA'), 'xyz123')
        if os.path.join(os.getenv('APPDATA'), 'xyz123'):
            testfile = testfile + str(n)
        else:
            file_exists = False
        if n > 10:
            file_exists = False
    return testfile


def test_file_creation():
    """Test initialization of StickySettings and file creation"""
    testfile = create_filename()
    _ = ss(testfile)
    assert os.path.isfile(testfile + '.json')
    os.remove(testfile + '.json')


def test_store_value():
    """Test creating a file and key value pair and getting the value"""
    testfile = create_filename()
    test_user = ss(testfile)
    test_user.new('test', True)
    assert test_user.get('test')
    os.remove(testfile + '.json')


def test_set_value():
    """Test setting a value of an existing key"""
    testfile = create_filename()
    test_user = ss(testfile)
    test_user.new('test', False)
    test_user.set('test', True)
    assert test_user.get('test')
    os.remove(testfile + '.json')


def test_set_value_failure():
    """Test failure when setting a value for a key that does not exist"""
    testfile = create_filename()
    test_user = ss(testfile)
    test_user.new('test', True)
    with pytest.raises(KeyError):
        test_user.set('Folder', 'AnyFolder')
    os.remove(testfile + '.json')


def test_get_value_failure():
    """Test failure when requesting a value for a key that does not exist"""
    testfile = create_filename()
    test_user = ss(testfile)
    test_user.new('test', True)
    with pytest.raises(KeyError):
        test_user.get('Folder')
    os.remove(testfile + '.json')
