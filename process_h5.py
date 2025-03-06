import h5py
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Path to the HDF5 file')
    args = parser.parse_args()

    with h5py.File(args.filename, 'r') as f:
        for key in f.keys():
            data = f[key][()]
            np.save(f'{key}.npy', data)
            print(f"Saved {key} with shape {data.shape}")
            
            print(f"Dataset: {key}")
            print(f"  Shape: {data.shape}")
            print(f"  Type: {data.dtype}")
            print(f"  Size: {data.size} elements")
            print()

if __name__ == '__main__':
    main()  