def get_enum_val_str(enum_val):
    val = str(enum_val).split(".")[1]
    if ":" in val:
        val = val.split(":")[0]
    return val