"""Speed-band offset, never a post-dispatch motion floor."""
import os
OFFSET=float(os.environ.get('UCR_OBSERVED_SPEED_OFFSET','4'))
assert 0<=OFFSET<=5 and OFFSET==OFFSET
def band(value):
    return value+OFFSET if value>0 else value
