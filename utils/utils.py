
def unique_ordered_list(seq):
    """Return a list of unique elements in the order they first appeared."""
    seen = set()
    unique_list = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list