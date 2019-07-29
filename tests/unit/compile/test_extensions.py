import pytest

from flambe.compile import extensions as exts


def test_download_extensions():
    extensions = {
            'ext': './local/file',
            'ext1': 'other/local/file',
            'ext2': 'other',
            'ext2': 'pypi==1.12.1',
    }

    ret = exts.download_extensions(extensions, None)
    for k, v in extensions.items():
        assert k in ret
        assert v == ret[k]


def test_is_installed_module():
    assert exts.is_installed_module("pytest") is True
    assert exts.is_installed_module("some_inexistent_package_0987654321") is False
