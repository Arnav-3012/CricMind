from cricmind import config


def test_split_years_are_ordered():
    assert config.TRAIN_END_YEAR < config.VAL_YEAR < config.TEST_START_YEAR <= config.TEST_END_YEAR
