RGB = [
    [0, 0, 0],  # background
    [128, 0, 0],  # pin
    [0, 128, 0],  # pin_w_solder
    [128, 128, 0],  # forceps
    [0, 0, 128],  # arm_glove
    [128, 0, 128],  # arm_wo_glove
]

BGR = [c[::-1] for c in RGB]
