import numpy as np


def remainderlessDividable(val, divider, ff):
    assert divider > 0
    assert ff == 0 or ff == 1
    return (val + divider * ff - val % divider)


def colorizeLabel(label, colors):
    colorized = np.zeros(label.shape[:2] + (3,), np.uint8)
    for classId, color in enumerate(colors):
        classMask = label == classId
        colorized[..., 0] += np.multiply(classMask, color[0], dtype=np.uint8, casting='unsafe')
        colorized[..., 1] += np.multiply(classMask, color[1], dtype=np.uint8, casting='unsafe')
        colorized[..., 2] += np.multiply(classMask, color[2], dtype=np.uint8, casting='unsafe')
    return colorized