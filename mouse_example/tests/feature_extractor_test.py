from mouse_example.MouseActions import MouseActionExtractor, MouseFeatureExtractor
from mouse_example.mouse_raw_data_acquisition import MouseEventHandler

try:
    """
    A window will be opened. Move and click the mouse and than close it. Raw data will be acquired.
    """
    meh = MouseEventHandler()
    raw_data_seq_acquired = meh.raw_data

    act_extractor = MouseActionExtractor()

    actions = act_extractor.extract_actions(raw_data_seq_acquired)

    features_extractor = MouseFeatureExtractor(actions)
    features_extractor.extract_features()

    print features_extractor.datasets

except Exception, e:
    raise e
