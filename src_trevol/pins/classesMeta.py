class_x_rgb = [
    ('background', [0, 0, 0]),
    ('pin', [128, 0, 0]),
    ('pin_w_solder', [0, 128, 0]),
    ('forceps', [128, 128, 0]),
    ('arm_glove', [0, 0, 128]),
    ('arm_wo_glove', [128, 0, 128]),
]

classNames = [x[0] for x in class_x_rgb]

RGB = [x[1] for x in class_x_rgb]

BGR = [c[::-1] for c in RGB]
