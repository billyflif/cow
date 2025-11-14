
COW_BODY_MEASUREMENTS = {
    # Ear Number: [Body Height, Body Length, Chest Girth, Cannon Circumference, Cross Height]
    "160": [135.5, 162, 204, 22.5, 134],
    "225": [132, 158, 190, 21, 134.5],
    "170": [127, 162, 198, 22, 133],
    "216": [133, 166, 192, 21, 139],
    "200": [125, 152, 197, 21, 125],
    "143": [130, 154, 199, 20, 130],
    "176": [131, 156, 196, 21, 134],
    "185": [139, 162, 214, 22, 138],
    "173": [130, 161, 210, 21, 134],
    "175": [125, 160, 201, 20, 125],
    "220": [135, 157, 194, 20, 139],
    "217": [138, 154, 188, 22, 138],
    "171": [126, 157, 193, 20, 126],
    "218": [132, 162, 198, 21, 131],
    "158": [134, 160, 191, 21, 132],
    "165": [132, 158, 172, 20, 131],
    "178": [135, 162, 209, 23, 136],
    "163": [131, 161, 198, 20, 135],
    "234": [130, 160, 185, 21, 132],
    "243": [138, 155, 197, 22, 138],
    "111": [138, 155, 197, 22, 138],#xuni
    "104": [125, 152, 197, 21, 125],#xuni 92zuidaa
    "236": [132, 158, 178, 22, 134],
    "240": [129, 165, 190, 22, 131],
    "167": [129, 157, 208, 21, 132],
    "244": [130, 151, 196, 20, 134],
    "205": [134, 153, 186, 22, 134],
    "222": [132, 155, 190, 21, 134],
    "226": [133, 161, 197, 22, 133],

}

MEASUREMENT_LABELS = [
    "Body Height",  # 体高
    "Body Length",  # 体斜长
    "Chest Girth",  # 胸围
    "Cannon Circ",  # 管围
    "Cross Height",  # 十字部高
]


def get_measurements(cow_id_name):
    """
    Get body measurements for a given cow ID

    Args:
        cow_id_name: Name of the cow (should match folder name in gallery)

    Returns:
        dict: Dictionary with measurement labels and values
    """
    if cow_id_name in COW_BODY_MEASUREMENTS:
        measurements = COW_BODY_MEASUREMENTS[cow_id_name]
        return {label: value for label, value in zip(MEASUREMENT_LABELS, measurements) if value is not None}
    return None