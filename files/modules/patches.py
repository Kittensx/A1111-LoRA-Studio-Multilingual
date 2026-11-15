from collections import defaultdict


def patch(key, obj, field, replacement):
    """Replaces a function in a module or a class.

    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).
    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.

    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
        replacement: the new function

    Returns:
        the original function
    """

    patch_key = (obj, field)
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")

    original_func = getattr(obj, field)
    stack = stacks[patch_key]
    if stack and original_func is not stack[-1][1]:
        raise RuntimeError(f"external patch owns {field}; refusing to overwrite it")
    owned = field in vars(obj)
    # A refused setattr must not leave a phantom owner in the registry.
    setattr(obj, field, replacement)
    originals[key][patch_key] = original_func
    stack.append((key, replacement, owned))

    return original_func


def undo(key, obj, field):
    """Undoes the peplacement by the patch().

    If the function is not replaced, raises an exception.

    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string

    Returns:
        Always None
    """

    patch_key = (obj, field)

    if patch_key not in originals[key]:
        raise RuntimeError(f"there is no patch for {field} to undo")

    stack = stacks[patch_key]
    if not stack or stack[-1][0] != key:
        raise RuntimeError(f"patch for {field} must be undone in reverse installation order")
    _, replacement, owned = stack[-1]
    if getattr(obj, field) is not replacement:
        raise RuntimeError(f"external patch owns {field}; refusing to remove it")
    original_func = originals[key].pop(patch_key)
    if owned:
        setattr(obj, field, original_func)
    else:
        delattr(obj, field)
    stack.pop()
    if not stack:
        del stacks[patch_key]

    return None


def original(key, obj, field):
    """Returns the original function for the patch created by the patch() function"""
    patch_key = (obj, field)

    return originals[key].get(patch_key, None)


def undo_many(key, targets):
    """Preflight a set of distinct hooks before removing any of them.

    Used for class/module attributes (as in LoRA); arbitrary custom descriptors
    which can fail during setattr/delattr are not a transactional storage API.
    """
    targets = list(targets)
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate patch target in undo_many")
    for obj, field in targets:
        patch_key = (obj, field)
        stack = stacks.get(patch_key, ())
        if patch_key not in originals.get(key, {}):
            raise RuntimeError(f"there is no patch for {field} to undo")
        if not stack or stack[-1][0] != key:
            raise RuntimeError(f"patch for {field} must be undone in reverse installation order")
        if getattr(obj, field) is not stack[-1][1]:
            raise RuntimeError(f"external patch owns {field}; refusing to remove it")
    for obj, field in targets:
        undo(key, obj, field)


originals = defaultdict(dict)
stacks = defaultdict(list)
