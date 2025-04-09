import h5py
from data.edf_data_import import read_mat_string

mat_file = ".../data/EDF_RawData.mat"

with h5py.File(mat_file, 'r') as f:
    # List the top-level keys in the MAT file
    print("Keys:", list(f.keys()))

    # Access the 'allData' group
    allData = f['allData']

    # Access the 'fileName' field: it is stored as an array of references.
    file_names_ds = allData['fileName']
    file_names_list = []
    for i in range(file_names_ds.shape[0]):
        ref = file_names_ds[i, 0]
        name_str = read_mat_string(ref, f)
        file_names_list.append(name_str)

    print("EDF File Names:")
    for name in file_names_list:
        print("  ", name)

    # Access header and record fields similarly:
    hdr = allData['hdr']
    record = allData['record']