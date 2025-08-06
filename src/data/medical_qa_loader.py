def create_federated_dataloaders(dataset, 
                                num_clients: int, 
                                batch_size: int = 8,
                                iid: bool = True,
                                alpha: float = 0.5) -> List[DataLoader]:
    """
    Create federated dataloaders by distributing data across clients
    
    Args:
        dataset: Dataset to split (can be Dataset or Subset)
        num_clients: Number of clients
        batch_size: Batch size for each client
        iid: Whether to use IID distribution
        alpha: Dirichlet parameter for non-IID distribution
        
    Returns:
        List of DataLoaders for each client
    """
    # Handle both regular Dataset and Subset objects
    if hasattr(dataset, 'data'):
        # Original dataset with .data attribute
        data_samples = dataset.data
        total_samples = len(data_samples)
    elif hasattr(dataset, 'dataset'):
        # Subset object - get the underlying dataset and indices
        original_dataset = dataset.dataset
        indices = dataset.indices
        total_samples = len(indices)
        
        # Create a list of samples based on the subset indices
        if hasattr(original_dataset, 'data'):
            data_samples = [original_dataset.data[i] for i in indices]
        else:
            # If original dataset doesn't have .data, use the subset directly
            data_samples = [dataset[i] for i in range(len(dataset))]
    else:
        # Fallback: iterate through the dataset
        data_samples = [dataset[i] for i in range(len(dataset))]
        total_samples = len(data_samples)
    
    print(f"Creating federated dataloaders for {total_samples} samples across {num_clients} clients")
    
    if total_samples < num_clients:
        raise ValueError(f"Not enough samples ({total_samples}) for {num_clients} clients")
    
    # Create client datasets
    if iid:
        # IID distribution: randomly shuffle and split evenly
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        # Calculate samples per client
        samples_per_client = total_samples // num_clients
        remainder = total_samples % num_clients
        
        client_datasets = []
        start_idx = 0
        
        for i in range(num_clients):
            # Distribute remainder samples to first few clients
            client_samples = samples_per_client + (1 if i < remainder else 0)
            end_idx = start_idx + client_samples
            
            client_indices = indices[start_idx:end_idx]
            
            # Create subset of data for this client
            if hasattr(dataset, 'dataset'):
                # Dataset is already a Subset, create nested subset
                original_indices = [dataset.indices[idx] for idx in client_indices]
                client_dataset = Subset(dataset.dataset, original_indices)
            else:
                # Dataset is original, create subset
                client_dataset = Subset(dataset, client_indices)
            
            client_datasets.append(client_dataset)
            start_idx = end_idx
            
    else:
        # Non-IID distribution using Dirichlet distribution
        print(f"Creating non-IID distribution with alpha={alpha}")
        
        # Extract labels for non-IID splitting
        labels = []
        for i in range(total_samples):
            try:
                if hasattr(dataset, 'dataset'):
                    # Subset object
                    sample = dataset[i]
                else:
                    # Regular dataset
                    sample = dataset[i]
                
                if isinstance(sample, dict) and 'labels' in sample:
                    # Take first label if it's a sequence
                    label = sample['labels']
                    if hasattr(label, '__len__') and not isinstance(label, str):
                        label = label[0] if len(label) > 0 else 0
                    labels.append(int(label) if label != -100 else 0)
                else:
                    # Fallback: assign random label
                    labels.append(i % 10)  # Assume 10 classes max
                    
            except Exception as e:
                print(f"Warning: Error extracting label for sample {i}: {e}")
                labels.append(0)  # Default label
        
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        print(f"Found {num_classes} unique labels: {unique_labels}")
        
        # Create Dirichlet distribution for each client
        client_datasets = []
        all_indices = np.arange(total_samples)
        
        for i in range(num_clients):
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_classes)
            
            # Calculate number of samples for each class for this client
            client_indices = []
            
            for class_idx, class_label in enumerate(unique_labels):
                class_indices = all_indices[labels == class_label]
                num_class_samples = int(proportions[class_idx] * len(class_indices))
                
                if num_class_samples > 0:
                    # Randomly select samples from this class
                    selected = np.random.choice(
                        class_indices, 
                        size=min(num_class_samples, len(class_indices)), 
                        replace=False
                    )
                    client_indices.extend(selected.tolist())
                    
                    # Remove selected indices from available pool
                    all_indices = np.setdiff1d(all_indices, selected)
                    labels = np.delete(labels, np.isin(all_indices, selected))
            
            # Ensure minimum samples per client
            min_samples = max(1, total_samples // (num_clients * 10))  # At least 10% of average
            if len(client_indices) < min_samples and len(all_indices) > 0:
                additional_needed = min(min_samples - len(client_indices), len(all_indices))
                additional_indices = np.random.choice(all_indices, size=additional_needed, replace=False)
                client_indices.extend(additional_indices.tolist())
                all_indices = np.setdiff1d(all_indices, additional_indices)
            
            # Create client dataset
            if len(client_indices) > 0:
                if hasattr(dataset, 'dataset'):
                    # Dataset is already a Subset
                    original_indices = [dataset.indices[idx] for idx in client_indices]
                    client_dataset = Subset(dataset.dataset, original_indices)
                else:
                    # Dataset is original
                    client_dataset = Subset(dataset, client_indices)
                
                client_datasets.append(client_dataset)
            
        # Distribute any remaining samples
        if len(all_indices) > 0:
            print(f"Distributing {len(all_indices)} remaining samples")
            for i, remaining_idx in enumerate(all_indices):
                client_idx = i % len(client_datasets)
                
                if hasattr(dataset, 'dataset'):
                    original_idx = dataset.indices[remaining_idx]
                    # Add to existing client dataset - this is tricky with Subset
                    # Create new Subset with additional index
                    existing_indices = list(client_datasets[client_idx].indices)
                    existing_indices.append(original_idx)
                    client_datasets[client_idx] = Subset(dataset.dataset, existing_indices)
                else:
                    existing_indices = list(client_datasets[client_idx].indices)
                    existing_indices.append(remaining_idx)
                    client_datasets[client_idx] = Subset(dataset, existing_indices)
    
    # Create DataLoaders
    dataloaders = []
    for i, client_dataset in enumerate(client_datasets):
        dataloader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False  # Keep all samples
        )
        dataloaders.append(dataloader)
        
        print(f"Client {i}: {len(client_dataset)} samples, {len(dataloader)} batches")
    
    print(f"Created {len(dataloaders)} federated dataloaders")
    return dataloaders
