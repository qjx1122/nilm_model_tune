import argparse
import h5py

p = argparse.ArgumentParser()
p.add_argument("--path", required=True)
args = p.parse_args()

with h5py.File(args.path, "r") as f:
    print("Root keys:")
    for k in f.keys():
        print(" ", k)
    print("\nIf this is a NILMTK HDF5 file, use NILMTK's DataSet API to inspect buildings/meters.")
    print("This script intentionally does not assume one fixed HDF5 internal layout.")
