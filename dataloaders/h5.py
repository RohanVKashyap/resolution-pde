import h5py
import numpy as np
import argparse
import os

import h5py
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    with h5py.File(args.filename, 'r') as f:
        print("Available keys in the HDF5 file:")
        for key in f.keys():
            print(f"Dataset: {key}")
            data_shape = f[key].shape
            data_dtype = f[key].dtype
            data_size = np.prod(data_shape)
            
            print(f"  Shape: {data_shape}")
            print(f"  Type: {data_dtype}")
            print(f"  Size: {data_size} elements")
            print(f"  Memory required: {data_size * np.dtype(data_dtype).itemsize / (1024**3):.2f} GiB")
            
            # For very large datasets like 'particles', process one sample at a time
            if key == "particles" or data_size * np.dtype(data_dtype).itemsize > 1 * (1024**3):  # If over 1 GiB
                # Process one sample at a time from the first dimension
                for i in range(data_shape[0]):
                    print(f"Processing {key} sample {i}")
                    
                    # For extremely large arrays, we may need to further subdivide
                    if len(data_shape) >= 3 and data_shape[1] > 100:
                        # Process chunks of the second dimension
                        chunk_size = 100  # Process 100 timesteps at once
                        for j in range(0, data_shape[1], chunk_size):
                            end_j = min(j + chunk_size, data_shape[1])
                            print(f"  Processing timesteps {j} to {end_j-1}")
                            
                            sub_chunk = f[key][i:i+1, j:end_j]
                            output_file = os.path.join(args.output_dir, f"{key}_sample_{i}_timesteps_{j}_{end_j-1}.npy")
                            np.save(output_file, sub_chunk)
                            print(f"    Saved to {output_file}")
                    else:
                        # Process the entire sample at once
                        try:
                            sample_data = f[key][i:i+1]
                            output_file = os.path.join(args.output_dir, f"{key}_sample_{i}.npy")
                            np.save(output_file, sample_data)
                            print(f"  Saved to {output_file}")
                        except Exception as e:
                            print(f"  Error saving {key} sample {i}: {e}")
            else:
                # Small dataset, save as is
                try:
                    data = f[key][()]
                    output_file = os.path.join(args.output_dir, f"{key}.npy")
                    np.save(output_file, data)
                    print(f"  Saved to {output_file}")
                except Exception as e:
                    print(f"  Error saving {key}: {e}")
            
            print()

if __name__ == '__main__':
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--filename', type=str, required=True, help='Path to the HDF5 file')
#     args = parser.parse_args()

#     with h5py.File(args.filename, 'r') as f:
#         for key in f.keys():
#             data = f[key][()]
#             np.save(f'{key}.npy', data)
#             print(f"Saved {key} with shape {data.shape}")
            
#             print(f"Dataset: {key}")
#             print(f"  Shape: {data.shape}")
#             print(f"  Type: {data.dtype}")
#             print(f"  Size: {data.size} elements")
#             print()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--filename', type=str, required=True, help='Path to the HDF5 file')
#     parser.add_argument('--chunk_size', type=int, default=100, help='Number of samples to process at once')
#     parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
#     args = parser.parse_args()
    
#     # Create output directory if it doesn't exist
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     with h5py.File(args.filename, 'r') as f:
#         print("Available keys in the HDF5 file:")
#         for key in f.keys():
#             print(f"Dataset: {key}")
#             data_shape = f[key].shape
#             data_dtype = f[key].dtype
#             data_size = np.prod(data_shape)
            
#             print(f"  Shape: {data_shape}")
#             print(f"  Type: {data_dtype}")
#             print(f"  Size: {data_size} elements")
#             print(f"  Memory required: {data_size * np.dtype(data_dtype).itemsize / (1024**3):.2f} GiB")
            
#             # Process in chunks
#             if len(data_shape) >= 2:
#                 # Determine chunk dimensions based on the first dimension
#                 total_samples = data_shape[0]
#                 chunk_size = min(args.chunk_size, total_samples)
                
#                 for i in range(0, total_samples, chunk_size):
#                     end_idx = min(i + chunk_size, total_samples)
#                     print(f"Processing {key} chunks {i} to {end_idx-1}")
                    
#                     # Load and save chunk
#                     chunk_data = f[key][i:end_idx]
#                     output_file = os.path.join(args.output_dir, f"{key}_chunk_{i}_{end_idx-1}.npy")
#                     np.save(output_file, chunk_data)
#                     print(f"  Saved chunk to {output_file}")
#             else:
#                 # Small dataset, save as is
#                 try:
#                     data = f[key][()]
#                     output_file = os.path.join(args.output_dir, f"{key}.npy")
#                     np.save(output_file, data)
#                     print(f"  Saved to {output_file}")
#                 except Exception as e:
#                     print(f"  Error saving {key}: {e}")
            
#             print()