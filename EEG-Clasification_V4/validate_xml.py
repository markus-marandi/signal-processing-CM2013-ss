import os
from scipy.io import loadmat


def matstruct_to_dict(mat_struct):
    """
    Recursively convert a MATLAB mat_struct object into a nested dictionary.
    This function handles simple structures (arrays, nested fields) that
    come from loadmat with struct_as_record=False.
    """
    # If the object does not have _fieldnames (is not a mat_struct), return it as-is.
    if not hasattr(mat_struct, "_fieldnames"):
        return mat_struct
    out_dict = {}
    for field in mat_struct._fieldnames:
        value = getattr(mat_struct, field)
        # Recursively convert nested mat_struct objects
        if hasattr(value, "_fieldnames"):
            out_dict[field] = matstruct_to_dict(value)
        elif isinstance(value, list):
            out_dict[field] = [matstruct_to_dict(item) if hasattr(item, "_fieldnames") else item for item in value]
        else:
            out_dict[field] = value
    return out_dict


def validate_structure(all_data):
    """
    Validates that each element in the all_data structure (converted to dict)
    has the expected keys:
      - fileName
      - events
      - stages
      - epochLength
      - annotation

    It prints a summary for each element.
    """
    expected_fields = ["fileName", "events", "stages", "epochLength", "annotation"]

    # If all_data is a numpy array of mat_struct objects, convert it to a list of dictionaries.
    if hasattr(all_data, "tolist"):
        all_data = all_data.tolist()

    # Convert each element if necessary.
    converted_data = []
    for elem in all_data:
        try:
            conv = matstruct_to_dict(elem)
        except Exception as e:
            conv = elem
        converted_data.append(conv)

    print("Validating structure for {} elements:\n".format(len(converted_data)))
    for idx, elem in enumerate(converted_data):
        print("Element {}:".format(idx + 1))
        missing = []
        for field in expected_fields:
            if field not in elem:
                missing.append(field)
            else:
                value = elem[field]
                if field == "fileName":
                    try:
                        value_str = str(value).strip()
                    except Exception:
                        value_str = str(value)
                    print("  {}: {} (type: {})".format(field, value_str, type(value).__name__))
                elif field == "events":
                    try:
                        num_events = len(value) if value is not None else "N/A"
                    except Exception:
                        num_events = "N/A"
                    print("  {}: {} events (type: {})".format(field, num_events, type(value).__name__))
                elif field == "stages":
                    try:
                        stages_len = len(value) if value is not None else "N/A"
                    except Exception:
                        stages_len = "N/A"
                    print("  {}: length {} (type: {})".format(field, stages_len, type(value).__name__))
                elif field == "epochLength":
                    print("  {}: {} (type: {})".format(field, value, type(value).__name__))
                elif field == "annotation":
                    print("  {}: {} (type: {})".format(field, value, type(value).__name__))
        if missing:
            print("  Missing fields:", missing)
        print("")


if __name__ == "__main__":
    # Update the path if necessary
    mat_file = "Input/XML_RawData.mat"

    if not os.path.exists(mat_file):
        print("MAT file not found at:", mat_file)
    else:
        # Load the MAT file; using squeeze_me=True helps simplify structure.
        mat_contents = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        print("Top-level keys in MAT file:", list(mat_contents.keys()))

        if "allData" not in mat_contents:
            print("'allData' variable not found in the MAT file.")
        else:
            all_data = mat_contents["allData"]
            validate_structure(all_data)