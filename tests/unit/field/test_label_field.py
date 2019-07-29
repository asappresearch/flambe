from flambe.field import LabelField
from flambe.tokenizer import LabelTokenizer


def test_tokenizer():
    """Test one hot featurizer."""
    tokenizer = LabelTokenizer()

    dummy = 'LABEL1'
    assert tokenizer.tokenize(dummy) == ['LABEL1']

    dummy = 'LABEL1,LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1,LABEL2']

    dummy = 'LABEL1 LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1 LABEL2']


def test_tokenizer_multilabel():
    """Test one hot featurizer."""
    tokenizer = LabelTokenizer(multilabel_sep=',')

    dummy = 'LABEL1'
    assert tokenizer.tokenize(dummy) == ['LABEL1']

    dummy = 'LABEL1,LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1', 'LABEL2']

    dummy = 'LABEL1 LABEL2'
    assert tokenizer.tokenize(dummy) == ['LABEL1 LABEL2']


def test_label_process():
    """Test label nuemricalization."""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField()
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert list(field.process('LABEL1')) == [0]
    assert list(field.process('LABEL2')) == [2]
    assert list(field.process('LABEL3')) == [1]


def test_label_process_multilabel():
    """Test label nuemricalization."""
    dummy = ['LABEL1,LABEL2', 'LABEL3', 'LABEL2,LABEL1', 'LABEL2']

    field = LabelField()
    field.setup(dummy)
    assert len(field.vocab) == 4

    field = LabelField(multilabel_sep=',')
    field.setup(dummy)
    assert len(field.vocab) == 3

    assert list(field.process('LABEL1,LABEL2')) == [0, 1]
    assert list(field.process('LABEL2,LABEL1')) == [1, 0]
    assert list(field.process('LABEL2')) == [1]
    assert list(field.process('LABEL3')) == [2]


def test_label_process_one_hot():
    """Test label nuemricalization."""
    dummy = ['LABEL1', 'LABEL3', 'LABEL2', 'LABEL2']

    field = LabelField(one_hot=True)
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert list(field.process('LABEL1')) == [1, 0, 0]
    assert list(field.process('LABEL2')) == [0, 0, 1]
    assert list(field.process('LABEL3')) == [0, 1, 0]



def test_label_process_multilabel_one_hot():
    """Test label nuemricalization."""
    dummy = ['LABEL1,LABEL2', 'LABEL3', 'LABEL2,LABEL1', 'LABEL2']

    field = LabelField(multilabel_sep=',', one_hot=True)
    field.setup(dummy)

    assert len(field.vocab) == 3
    assert list(field.process('LABEL1,LABEL2')) == [1, 1, 0]
    assert list(field.process('LABEL2,LABEL1')) == [1, 1, 0]
    assert list(field.process('LABEL2')) == [0, 1, 0]
    assert list(field.process('LABEL3')) == [0, 0, 1]
