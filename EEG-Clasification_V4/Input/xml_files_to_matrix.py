import os
import glob
from xml.dom import minidom
from scipy.io import savemat

def read_xml(xmlfile):
    """
    Reads an XML file and extracts:
      - epoch_length: from the <EpochLength> element (float)
      - events: list of event dictionaries (for non-sleep-stage events)
      - stages: list of integers representing sleep stage annotations (based on duration)
      - annotation: 1 if at least one <ScoredEvent> exists, otherwise 0

    The sleep stage mapping is as follows:
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
        raise Exception("Failed to read XML file {}: {}".format(xmlfile, e))

    # Get epoch length from the <EpochLength> element.
    epoch_elements = doc.getElementsByTagName("EpochLength")
    if epoch_elements.length > 0 and epoch_elements[0].firstChild:
        epoch_length = float(epoch_elements[0].firstChild.data.strip())
    else:
        epoch_length = None

    # Get all <ScoredEvent> elements.
    events_elements = doc.getElementsByTagName("ScoredEvent")
    if events_elements.length > 0:
        annotation = 1
    else:
        annotation = 0

    events_vector = []
    stages = []

    # Define sleep stage concepts with their mapping.
    stages_concept = [
        "SDO:NonRapidEyeMovementSleep-N1",  # mapped to 4
        "SDO:NonRapidEyeMovementSleep-N2",  # mapped to 3
        "SDO:NonRapidEyeMovementSleep-N3",  # mapped to 2
        "SDO:NonRapidEyeMovementSleep-N4",  # mapped to 1
        "SDO:RapidEyeMovementSleep",         # mapped to 0
        "SDO:WakeState"                      # mapped to 5
    ]

    # Process each ScoredEvent element.
    for i in range(events_elements.length):
        event_elem = events_elements[i]

        # Extract the EventConcept.
        event_concept_elements = event_elem.getElementsByTagName("EventConcept")
        if event_concept_elements.length == 0 or not event_concept_elements[0].firstChild:
            annotation = 0
            break  # If no event concept, skip further processing.
        name = event_concept_elements[0].firstChild.data.strip()

        # Extract Start and Duration (using defaults if missing).
        start = 0.0
        duration = 0
        start_elements = event_elem.getElementsByTagName("Start")
        if start_elements.length > 0 and start_elements[0].firstChild:
            start = float(start_elements[0].firstChild.data.strip())
        duration_elements = event_elem.getElementsByTagName("Duration")
        if duration_elements.length > 0 and duration_elements[0].firstChild:
            duration = int(float(duration_elements[0].firstChild.data.strip()))

        # Extract optional fields: Desaturation, SpO2Nadir, and Text.
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

        # Determine if the event represents a known sleep stage.
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
    Processes all XML files (matching R*.xml) in the specified directory and returns a list
    of dictionaries. Each dictionary contains:
        - fileName: the XML file name,
        - events: list of non-stage events,
        - stages: list of sleep stage annotations,
        - epochLength: the epoch length (float),
        - annotation: flag indicating whether events were found (1) or not (0).
    """
    xml_pattern = os.path.join(xml_directory, "R*.xml")
    xml_files = sorted(glob.glob(xml_pattern))

    results = []
    for xml_file in xml_files:
        try:
            events, stages, epoch_length, annotation = read_xml(xml_file)
        except Exception as e:
            print("Error processing {}: {}".format(xml_file, e))
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
    Saves the provided data (a list of dictionaries) as a MATLAB MAT file.
    The output MAT file will have one top-level variable: 'allData'.
    """
    mat_data = {"allData": all_data}
    savemat(output_file, mat_data)
    print("Saved MAT file to:", output_file)


if __name__ == "__main__":
    # Define the directory containing your XML files.
    xml_directory = "/Users/markus/university/signal-processing-CM2013-ss/data"

    # Process all XML files in the directory.
    xml_data_structure = process_all_xml(xml_directory)
    print("Processed {} XML files.".format(len(xml_data_structure)))
    for data in xml_data_structure:
        print("File:", data["fileName"])
        print("  Epoch Length:", data["epochLength"])
        print("  Annotation:", data["annotation"])
        print("  Number of non-stage events:", len(data["events"]))
        print("  Stages length:", len(data["stages"]))
        print("")

    # Save the resulting structure to a MAT file in the same directory.
    output_mat_file = os.path.join(xml_directory, "XML_RawData.mat")
    save_to_mat(xml_data_structure, output_mat_file)