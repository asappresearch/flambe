import pytest
import numpy as np
import torch
from flambe.dataset import TabularDataset
from flambe.field import Field, TextField, LabelField


@pytest.fixture
def train_dataset():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/dummy_tabular/train.csv', sep=',')


@pytest.fixture
def train_dataset_no_header():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/no_header_dataset.csv', sep=',',
                                    header=None)


@pytest.fixture
def train_dataset_reversed():
    """Dummy dataset from file"""
    return TabularDataset.from_path('tests/data/dummy_tabular/train.csv', sep=',',
                                    columns=['label', 'text'])


@pytest.fixture
def full_dataset():
    """Dummy dataset from file"""
    return TabularDataset.from_path(train_path='tests/data/dummy_tabular/train.csv',
                                    val_path='tests/data/dummy_tabular/val.csv', sep=',')


@pytest.fixture
def dir_dataset():
    """Dummy dataset from directory"""
    return TabularDataset.from_path('tests/data/dummy_tabular', sep=',')


def test_valid_dataset():
    """Test trivial dataset build process"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 5, 5.5))
    val = (("ipsum quia dolor sit", 10, 3.5),)
    test = (("Ut enim ad minima veniam", 100, 35),)

    t = TabularDataset(train, val, test)

    assert len(t) == 4
    assert len(t.train) == 2
    assert len(t.val) == 1
    assert len(t.test) == 1

    def check(d, t):
        for i, tu in enumerate(d):
            v0, v1, v2 = tu
            assert t[i][0] == v0
            assert t[i][1] == v1
            assert t[i][2] == v2

    check(train, t.train)
    check(val, t.val)
    check(test, t.test)


def test_invalid_dataset():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 5.5))
    with pytest.raises(ValueError):
        TabularDataset(train)


def test_invalid_dataset2():
    """Test dataset is invalid as different splits contain different
    columns
    """
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 4, 5.5))
    val = (("ipsum quia dolor sit", 3.5),)
    with pytest.raises(ValueError):
        TabularDataset(train, val)


def test_incomplete_dataset():
    """Test dataset missing either val or test"""
    train = (("Lorem ipsum dolor sit amet", 3, 4.5),
             ("Sed ut perspiciatis unde", 4, 5.5))
    t = TabularDataset(train)

    assert len(t.val) == 0
    assert len(t.test) == 0


def test_cache_dataset():
    """Test caching the dataset"""
    train = (
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5),
            ("Lorem ipsum dolor sit amet", 3, 4.5),
            ("Sed ut perspiciatis unde", 5, 5.5))

    t = TabularDataset(train, cache=True)

    assert len(t.train.cached_data) == 0
    for i, _ in enumerate(t.train):
        assert len(t.train.cached_data) == i + 1


def test_column_attr(train_dataset):
    assert len(train_dataset.named_columns) == 2
    assert train_dataset.named_columns == ['text', 'label']


def test_column_attr2(train_dataset_no_header):
    assert train_dataset_no_header.named_columns is None


def test_named_columns():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3),
             ("Sed ut perspiciatis unde", 5.5))
    TabularDataset(train, named_columns=['col1', 'col2'])


def test_invalid_columns():
    """Test dataset is invalid as it has different columns"""
    train = (("Lorem ipsum dolor sit amet", 3),
             ("Sed ut perspiciatis unde", 5.5))
    with pytest.raises(ValueError):
        TabularDataset(train, named_columns=['some_random_col'])


def test_dataset_from_file(train_dataset):
    """Test loading a dataset from file"""
    dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    assert train_dataset[0][0] == dummy
    assert train_dataset[0][1] == '6'


def test_dataset_from_file_reversed(train_dataset_reversed):
    """Test loading a dataset from file"""
    dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    assert train_dataset_reversed[0][0] == '6'
    assert train_dataset_reversed[0][1] == dummy


def test_full_dataset_from_file(full_dataset):
    """Test loading a dataset from file"""
    train_dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    val_dummy = "malesuada. Integer id magna et ipsum cursus vestibulum. Mauris magna."

    assert full_dataset.train[0][0] == train_dummy
    assert full_dataset.train[0][1] == '6'

    assert full_dataset.val[0][0] == val_dummy
    assert full_dataset.val[0][1] == '8'

    assert full_dataset[0][0] == train_dummy
    assert full_dataset[100][0] == val_dummy


def test_dataset_from_dir(dir_dataset):
    """Test loading multiple datasets at once"""
    dummy = "justo. Praesent luctus. Curabitur egestas nunc sed libero. Proin sed"
    assert dir_dataset[0][0] == dummy
    assert dir_dataset[0][1] == '6'

    dummy = "malesuada. Integer id magna et ipsum cursus vestibulum. Mauris magna."
    assert dir_dataset[100][0] == dummy
    assert dir_dataset[100][1] == '8'


def test_dataset_length(train_dataset, full_dataset):
    """Test dataset length."""
    assert len(train_dataset) == 100
    assert len(full_dataset) == 200
    assert len(full_dataset.train) == 100
    assert len(full_dataset.val) == 100


def test_dataset_iter(train_dataset):
    """Test that the dataset is iterable."""
    for i, ex in enumerate(train_dataset):
        assert np.array_equal(ex, train_dataset[i])


def test_dataset_setitem(train_dataset):
    """Test dataset is immutable"""
    with pytest.raises(Exception):
        train_dataset[0] = 0


def test_dataset_deltitem(train_dataset):
    """Test dataset is immutable"""
    with pytest.raises(Exception):
        del train_dataset[0]


def test_dataset_transform():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": TextField(),
        "label": LabelField()
    }

    t = TabularDataset(train, transform=transform)

    assert hasattr(t, "text")
    assert hasattr(t, "label")

    assert t.label.vocab_size == 2
    assert t.text.vocab_size == 11


def test_dataset_transform_2():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": {
            "field": TextField()
        },
        "label": {
            "field": LabelField(),
            "columns": 1
        }
    }

    t = TabularDataset(train, transform=transform)

    assert hasattr(t, "text")
    assert hasattr(t, "label")

    assert t.label.vocab_size == 2


def test_dataset_transform_3():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "text": {
            "columns": 0
        },
        "label": {
            "field": LabelField(),
            "columns": 1
        }
    }

    with pytest.raises(ValueError):
        TabularDataset(train, transform=transform)


def test_dataset_transform_4():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "t1": {
            "field": TextField(),
            "columns": 1
        },
        "t2": {
            "field": TextField(),
            "columns": 1
        }
    }

    t = TabularDataset(train, transform=transform)

    assert t.train.cols() == 2


def test_dataset_transform_5():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "t1": {
            "field": TextField(),
            "columns": 0
        },
        "t2": {
            "field": TextField(),
            "columns": 0
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 2


def test_dataset_transform_6():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": [0, 1]
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 1


def test_dataset_transform_7():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other2": {
            "field": LabelField(),
            "columns": 0
        }
    }

    t = TabularDataset(train, transform=transform)
    assert t.train.cols() == 3


def test_dataset_transform_8():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": [0, 1]
        }
    }

    with pytest.raises(TypeError):
        t = TabularDataset(train, transform=transform)
        t.train.cols()


def test_dataset_transform_with_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": 'label'
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert len(t.train[0]) == 1


def test_dataset_transform_with_invalid_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "tx": {
            "field": LabelField(),
            "columns": 'none_existent'
        }
    }

    with pytest.raises(ValueError):
        TabularDataset(train, transform=transform, named_columns=['text', 'label'])


def test_dataset_transform_with_mixed_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    transform = {
        "label": {
            "field": LabelField(),
            "columns": 1,
        },
        "text": {
            "field": TextField(),
            "columns": 'text',
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert len(t.train) == 2
    assert len(t.train[0]) == 2


def test_dataset_transform_mixed_multiple_named_cols():
    train = (
            ("Lorem ipsum dolor sit amet", "POSITIVE"),
            ("Sed ut perspiciatis unde", "NEGATIVE"))

    class DummyField(Field):
        def setup(self, *data: np.ndarray) -> None:
            pass

        def process(self, ex1, ex2):
            return torch.tensor(0)

    transform = {
        "text": {
            "field": DummyField(),
            "columns": ['text', 'label']
        },
        "other": {
            "field": DummyField(),
            "columns": [0, 1]
        },
        "other2": {
            "field": DummyField(),
            "columns": [0, 'label']
        }
    }

    t = TabularDataset(train, transform=transform, named_columns=['text', 'label'])
    assert t.train.cols() == 3
