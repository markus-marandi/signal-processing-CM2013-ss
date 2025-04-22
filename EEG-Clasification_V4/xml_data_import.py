"""
xml_data_import.py

This module provides utility functions to load and process XML data stored
in a MATLAB v7.3 MAT file. It includes functions to:

  - Read individual XML files and extract sleep stage annotations,
    event information, and other metadata.
  - Process all XML files in a directory (matching R*.xml) and convert them
    into a list of dictionaries.
  - Convert the list of dictionaries into a MATLAB‐compatible structure
    and save it as a MAT file.
  - Load and describe the processed MAT file containing XML data.
"""

import os
import glob
import numpy as np
from xml.dom import minidom
from scipy.io import loadmat, savemat

def read_xml(xmlfile):
    """
    Reads an XML file and extracts:
      - epoch_length: from the <EpochLength> element (float)
      - events: list of event dictionaries (for non-sleep-stage events)
      - stages: list of integers representing sleep stage annotations
      - annotation: 1 if at least one <ScoredEvent> exists, otherwise 0

    Sleep stage mapping:
      SDO:NonRapidEyeMovementSleep-N1 -> 4,
      SDO:NonRapidEyeMovementSleep-N2 -> 3,
      SDO:NonRapidEyeMovementSleep-N3 -> 2,
      SDO:NonRapidEyeMovementSleep-N4 -> 1,
      SDO:RapidEyeMovementSleep        -> 0,
      SDO:WakeState                    -> 5.
    """
    try:
        doc = minidom.parse(xmlfile)
    except Exception as e:
        raise Exception(f"Failed to read XML file {xmlfile}: {e}")

    # Retrieve epoch length
    epoch_elements = doc.getElementsByTagName("EpochLength")
    if epoch_elements.length > 0 and epoch_elements[0].firstChild:
        epoch_length = float(epoch_elements[0].firstChild.data.strip())
    else:
        epoch_length = None

    # Process ScoredEvent elements
    events_elements = doc.getElementsByTagName("ScoredEvent")
    annotation = 1 if events_elements.length > 0 else 0

    events_vector = []
    stages = []

    # Sleep stage mapping (adjust as needed)
    stages_concept = [
        "SDO:NonRapidEyeMovementSleep-N1",  # mapped to 4
        "SDO:NonRapidEyeMovementSleep-N2",  # mapped to 3
        "SDO:NonRapidEyeMovementSleep-N3",  # mapped to 2
        "SDO:NonRapidEyeMovementSleep-N4",  # mapped to 1
        "SDO:RapidEyeMovementSleep",        # mapped to 0
        "SDO:WakeState"                     # mapped to 5
    ]

    for i in range(events_elements.length):
        event_elem = events_elements[i]

        # Get EventConcept
        event_concept_elements = event_elem.getElementsByTagName("EventConcept")
        if event_concept_elements.length == 0 or not event_concept_elements[0].firstChild:
            annotation = 0
            break  # Skip further processing if missing concept.
        name = event_concept_elements[0].firstChild.data.strip()

        # Extract Start and Duration, use defaults if missing.
        start = 0.0
        duration = 0
        start_elements = event_elem.getElementsByTagName("Start")
        if start_elements.length > 0 and start_elements[0].firstChild:
            start = float(start_elements[0].firstChild.data.strip())
        duration_elements = event_elem.getElementsByTagName("Duration")
        if duration_elements.length > 0 and duration_elements[0].firstChild:
            duration = int(float(duration_elements[0].firstChild.data.strip()))

        # Optional additional fields.
        baseline = 0.0
        nadir = 0.0
        text = ""
        desat_elements = event_elem.getElementsByTagName("Desaturation")
        if desat_elements.length > 0 and desat_elements[0].firstChild:
            baseline = float(desat_elements[0].firstChild.data.strip())
        nadir_elements = event_elem.getElementsByTagName("SpO2Nadir")
        if nadir_elements.length > 0 and nadir_elements[0].firstChild:
            nadir = float(nadir_elements[0].firstChild.data.strip())
        text_elements = event_elem.getElementsByTagName("Text")
        if text_elements.length > 0 and text_elements[0].firstChild:
            text = text_elements[0].firstChild.data.strip()

        event_dict = {
            "EventConcept": name,
            "Start": start,
            "Duration": duration,
            "Desaturation": baseline,
            "SpO2Nadir": nadir,
            "Text": text
        }

        # Map event to sleep stages (extend stage list by event duration)
        if name == stages_concept[0]:
            stages.extend([4] * duration)
        elif name == stages_concept[1]:
            stages.extend([3] * duration)
        elif name == stages_concept[2]:
            stages.extend([2] * duration)
        elif name == stages_concept[3]:
            stages.extend([1] * duration)
        elif name == stages_concept[4]:
            stages.extend([0] * duration)
        elif name == stages_concept[5]:
            stages.extend([5] * duration)
        else:
            events_vector.append(event_dict)

    return events_vector, stages, epoch_length, annotation


def process_all_xml(xml_directory):
    """
    Processes all XML files matching 'R*.xml' in the specified directory.

    Returns:
      A list of dictionaries, each with:
        - fileName: XML file name,
        - events: list of non-stage events,
        - stages: list of sleep stage annotations,
        - epochLength: epoch length (float),
        - annotation: 1 if events found, else 0.
    """
    xml_pattern = os.path.join(xml_directory, "R*.xml")
    xml_files = sorted(glob.glob(xml_pattern))

    results = []
    for xml_file in xml_files:
        try:
            events, stages, epoch_length, annotation = read_xml(xml_file)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue

        file_data = {
            "fileName": os.path.basename(xml_file),
            "events": events,
            "stages": stages,
            "epochLength": epoch_length,
            "annotation": annotation
        }
        results.append(file_data)
    return results


def save_to_mat(all_data, output_file):
    """
    Saves the provided all_data (list of dictionaries) to a MATLAB MAT file.
    Converts the list of dictionaries into a dictionary of lists for savemat.
    """
    converted = list_of_dicts_to_dict_of_lists(all_data)
    savemat(output_file, {"allData": converted})
    print("Saved MAT file to:", output_file)


# --- Helper functions for MATLAB mat_struct conversion and MAT file loading ---
def matstruct_to_dict(mat_struct):
    """
    Recursively converts a MATLAB mat_struct object (or numpy array of mat_struct)
    into a Python dictionary or list.
    """
    # If it's a numpy array, convert each element.
    if isinstance(mat_struct, np.ndarray):
        return [matstruct_to_dict(item) for item in mat_struct]
    # If it does not have the _fieldnames attribute, return it directly.
    if not hasattr(mat_struct, "_fieldnames"):
        return mat_struct
    out_dict = {}
    for field in mat_struct._fieldnames:
        value = getattr(mat_struct, field)
        out_dict[field] = matstruct_to_dict(value)
    return out_dict


def read_mat_string(ref, f) -> str:
    """
    Given an h5py reference and an open file f, decode the MATLAB string.
    """
    ds = f[ref]
    data = ds[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, np.ndarray):
        try:
            return data.tobytes().decode("utf-8").strip()
        except Exception as e:
            return str(data)
    else:
        return str(data)


def recursive_print(name: str, obj, f, indent: int = 0) -> None:
    """
    Recursively prints the HDF5 structure of a loaded MAT file.
    """
    spacer = "  " * indent
    if isinstance(obj, np.ndarray):
        print(f"{spacer}{name} (Array): shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, f.__class__):  # if it is a group
        print(f"{spacer}{name} (Group)")
        for key, item in obj.items():
            recursive_print(key, item, f, indent + 1)
    else:
        print(f"{spacer}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")


def load_xml_mat(mat_file: str) -> list:
    """
    Loads the MATLAB v7.3 MAT file containing XML data and returns the 'allData'
    variable as a list of dictionaries.

    Parameters:
        mat_file: Path to the MAT file.

    Returns:
        List of dictionaries, where each dictionary represents one XML file's data.
    """
    if not os.path.exists(mat_file):
        raise FileNotFoundError("MAT file not found at: " + mat_file)
    mat_contents = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    if "allData" not in mat_contents:
        raise ValueError("'allData' variable not found in the MAT file.")
    all_data = mat_contents["allData"]
    if hasattr(all_data, "tolist"):
        all_data = all_data.tolist()
    # Use the updated converter on each top-level element.
    converted = [matstruct_to_dict(elem) for elem in all_data]
    return converted


def describe_xml_data_structure(all_data: list) -> None:
    """
    Prints a summary description of the XML data structure loaded from MAT.
    """
    print("Keys in MAT file:")
    print(" - allData\n")
    print("Structure of 'allData':")
    print("  Attribute MATLAB_class: b'struct'")
    print("  Attribute MATLAB_fields: [array([...], dtype='|S1'), array([...], dtype='|S1'), array([...], dtype='|S1')]\n")
    print("Fields in allData:")
    if all_data:
        for field in all_data[0].keys():
            print(f"  Field: {field}")
    else:
        print("  (allData is empty)")
    print("\nDecoded file names from 'allData':")
    for idx, elem in enumerate(all_data):
        file_name = elem.get("fileName", "N/A")
        print(f"  {idx + 1}: {file_name}")
    print(f"\nValidating structure for {len(all_data)} elements:")
    expected_fields = ["fileName", "events", "stages", "epochLength", "annotation"]
    for idx, elem in enumerate(all_data):
        print(f"Element {idx + 1}:")
        if not isinstance(elem, dict):
            print(f"  (Non-dict element of type {type(elem).__name__}): {elem}\n")
            continue
        for field in expected_fields:
            value = elem.get(field, None)
            if field == "fileName":
                print(f"  {field}: {value} (type: {type(value).__name__})")
            elif field == "events":
                try:
                    num_events = len(value) if value is not None else "N/A"
                except Exception:
                    num_events = "N/A"
                print(f"  {field}: {num_events} events (type: {type(value).__name__})")
            elif field == "stages":
                try:
                    stages_len = len(value) if value is not None else "N/A"
                except Exception:
                    stages_len = "N/A"
                print(f"  {field}: length {stages_len} (type: {type(value).__name__})")
            elif field in ["epochLength", "annotation"]:
                print(f"  {field}: {value} (type: {type(value).__name__})")
        print("")


def get_xml_file_data(all_data: list, file_index: int) -> dict:
    """
    Returns the data for the XML file at the given 0-based index.
    """
    if file_index < 0 or file_index >= len(all_data):
        raise IndexError("File index out of bounds.")
    return all_data[file_index]


def get_field_value(xml_file_data: dict, field_name: str):
    """
    Retrieves the value for the given field from a single XML file's data.
    """
    return xml_file_data.get(field_name, None)


def slice_matrix(matrix: np.ndarray, axis: int, index: int) -> np.ndarray:
    """
    Extracts a slice from a numeric matrix along the specified axis.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if index < 0 or index >= matrix.shape[axis]:
        raise IndexError("Index out of range for the given axis.")
    return np.take(matrix, index, axis=axis)


def list_of_dicts_to_dict_of_lists(ld: list) -> dict:
    """
    Converts a list of dictionaries into a dictionary of lists, which is friendly for savemat.
    """
    if not ld:
        return {}
    keys = ld[0].keys()
    result = {key: [] for key in keys}
    for d in ld:
        for key in keys:
            result[key].append(d.get(key, None))
    return result


def save_to_mat(all_data: list, output_file: str) -> None:
    """
    Saves the all_data structure (list of dictionaries) into a MATLAB MAT file.
    """
    converted = list_of_dicts_to_dict_of_lists(all_data)
    savemat(output_file, {"allData": converted})
    print("Saved MAT file to:", output_file)


# Example usage when running the module directly:
if __name__ == "__main__":
    xml_directory = "Input"
    xml_data_structure = process_all_xml(xml_directory)
    print(f"Processed {len(xml_data_structure)} XML files.")
    for data in xml_data_structure:
        print("File:", data["fileName"])
        print("  Epoch Length:", data["epochLength"])
        print("  Annotation:", data["annotation"])
        print("  Number of non-stage events:", len(data["events"]))
        print("  Stages length:", len(data["stages"]))
        print("")

    output_mat_file = os.path.join(xml_directory, "XML_RawData.mat")
    save_to_mat(xml_data_structure, output_mat_file)