from flambe.nlp.language_modeling import PTBDataset


def test_dataset_ptb():
    dataset = PTBDataset()
    assert len(dataset.train) == 42068
    assert len(dataset.train[0]) == 1

    assert len(dataset.val) == 3370
    assert len(dataset.val[0]) == 1

    assert len(dataset.test) == 3761
    assert len(dataset.test[0]) == 1
