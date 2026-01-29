import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import random


#########################################################
# This is used for multi-resolution training (1D Data)
#########################################################

def multires_collate_fn(batch):
    """
    Custom collate function for multi-resolution data.
    Returns lists instead of stacked tensors to handle variable spatial dimensions.
    """
    if len(batch[0]) == 2:  # (x, y) pairs
        batch_x = [item[0] for item in batch]
        batch_y = [item[1] for item in batch]
        return batch_x, batch_y
    elif len(batch[0]) == 3:  # (x, y, coords) triplets
        batch_x = [item[0] for item in batch]
        batch_y = [item[1] for item in batch]
        batch_coords = [item[2] for item in batch]
        return batch_x, batch_y, batch_coords
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

class ResolutionGroupedSampler(Sampler):
    """
    Groups samples by spatial resolution for efficient batching.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by spatial resolution
        self.resolution_groups = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample_x, sample_y = dataset[idx]
            spatial_size = sample_x.shape[-1]  # Get spatial dimension
            self.resolution_groups[spatial_size].append(idx)
        
        print(f"Resolution groups found:")
        for res, indices in self.resolution_groups.items():
            print(f"  Resolution {res}: {len(indices)} samples")
    
    def __iter__(self):
        # Create batches within each resolution group
        batches = []
        
        for resolution, indices in self.resolution_groups.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches of same resolution
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)
        
        # Shuffle the order of batches (but not within batches)
        if self.shuffle:
            random.shuffle(batches)
        
        # Yield indices
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)


class ResolutionGroupedDataLoader:
    """
    DataLoader that groups samples by resolution for efficient batching.
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group samples by resolution
        self.resolution_groups = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample_x, sample_y = dataset[idx]
            spatial_size = sample_x.shape[-1]
            self.resolution_groups[spatial_size].append((sample_x, sample_y))
        
        print(f"Created resolution groups:")
        for res, samples in self.resolution_groups.items():
            print(f"  Resolution {res}: {len(samples)} samples")
        
        # Create separate DataLoaders for each resolution
        self.resolution_loaders = {}
        for resolution, samples in self.resolution_groups.items():
            # Create a simple dataset from the grouped samples
            res_dataset = SimpleDataset(samples)
            self.resolution_loaders[resolution] = DataLoader(
                res_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers
            )
    
    def __iter__(self):
        # Create iterators for each resolution
        iterators = {res: iter(loader) for res, loader in self.resolution_loaders.items()}
        
        # Get all batches from all resolutions
        all_batches = []
        for resolution, iterator in iterators.items():
            try:
                while True:
                    batch = next(iterator)
                    all_batches.append((resolution, batch))
            except StopIteration:
                pass
        
        # Shuffle the order of batches if needed
        if self.shuffle:
            random.shuffle(all_batches)
        
        # Yield batches
        for resolution, batch in all_batches:
            yield batch
    
    def __len__(self):
        return sum(len(loader) for loader in self.resolution_loaders.values())


class SimpleDataset(Dataset):
    """Simple dataset wrapper for pre-grouped samples."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_grouped_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """Create resolution-grouped dataloaders."""
    
    train_loader = ResolutionGroupedDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = ResolutionGroupedDataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = ResolutionGroupedDataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader