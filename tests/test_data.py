import pytest

from src.data.load_data import load_worldvqa_dataset


@pytest.mark.skip(reason="Requires network access and Hugging Face datasets")
def test_load_worldvqa_dataset_has_train_split():
    ds = load_worldvqa_dataset()
    assert "train" in ds
