import sys
import os
import torch
from lipMIP.neural_nets import data_loaders as data_loaders
from lipMIP.neural_nets import train as train
from lipMIP.relu_nets import ReLUNet

def main():
    """
    This script creates and trains a neural network, then saves the trained network,
    datasets, and dataloaders for future use in a specified directory.
    """
    # --- 1. Handle Command-Line Argument and Create Directories ---
    if len(sys.argv) < 2:
        print("Error: Please provide a name for the save directory.")
        print("Usage: python your_script_name.py <save_directory_name>")
        sys.exit(1) # Exit if no argument is provided

    save_name = sys.argv[1]
    base_save_dir = "random_networks"
    specific_save_dir = os.path.join(base_save_dir, save_name)

    # Create the directory structure (e.g., saved_networks/my_experiment)
    # The `exist_ok=True` flag prevents an error if the directory already exists.
    os.makedirs(specific_save_dir, exist_ok=True)
    print(f"--- Output will be saved in: '{specific_save_dir}' ---")

    # 2. Create the Dataset
    print("\nCreating synthetic dataset...")
    data_params = data_loaders.RandomKParameters(
        num_points=1024,
        dimension=2,
        num_classes=2,
        k=10,
        radius=0.01
    )
    random_dataset = data_loaders.RandomDataset(data_params, random_seed=1234)
    random_train, random_val = random_dataset.split_train_val(0.75)
    print("Dataset created and split.")

    # 3. Create the DataLoaders
    train_loader = torch.utils.data.DataLoader(random_train, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(random_val, batch_size=64, shuffle=False)
    print("DataLoaders are ready.")

    # 4. Define the Neural Network
    print("\nDefining the neural network...")
    network = ReLUNet([2, 16, 16, 16, 2])
    print("Network created.")

    # 5. Set up and run the training loop
    print("\nStarting training...")
    train_params = train.TrainParameters(
        random_train,
        random_val,
        num_epochs=500,
        test_after_epoch=100
    )
    train.training_loop(network, train_params)
    print("Training complete.")

    # 6. Save the network, datasets, and dataloaders
    print("\nSaving network, datasets, and dataloaders...")

    # Define full paths for the files to be saved
    network_path = os.path.join(specific_save_dir, 'trained_network.pth')
    datasets_path = os.path.join(specific_save_dir, 'datasets.pth')
    dataloaders_path = os.path.join(specific_save_dir, 'dataloaders.pth')

    network_list = ['random', network]

    # Save the trained network
    torch.save(network_list, network_path)

    # Save the datasets
    datasets = {
        'train': random_train,
        'validation': random_val
    }
    torch.save(datasets, datasets_path)
    
    # # Save the dataloaders
    # dataloaders = {
    #     'train_loader': train_loader,
    #     'val_loader': val_loader
    # }
    # torch.save(dataloaders, dataloaders_path)

    print("\nScript finished. All files have been saved:")
    print(f"- {network_path}")
    print(f"- {datasets_path}")
    # print(f"- {dataloaders_path}")

if __name__ == '__main__':
    main()