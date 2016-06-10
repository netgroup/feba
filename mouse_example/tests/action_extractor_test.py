from mouse_example.mouse_raw_data_acquisition import MouseEventHandler
from mouse_example.MouseActions import MouseActionExtractor

"""
A window will be opened. Move and click the mouse and than close it. Raw data will be acquired.
"""

meh = MouseEventHandler()
raw_data_seq_acquired = meh.raw_data

try:
    act_extractor = MouseActionExtractor()

    actions_extracted = act_extractor.extract_actions(raw_data_seq_acquired)
    print "Sequence of extracted actions: %s" % actions_extracted

    print "This is the end, my only friend"

except Exception, e:
    raise e