def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dict for MLflow logging.
    E.g. {"a": {"b": 1}} becomes {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
