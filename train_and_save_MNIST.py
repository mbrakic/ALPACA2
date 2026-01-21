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

    # 2. load the dataset
    train_loader = data_loaders.load_mnist_data('train', batch_size = 128, shuffle = True) 
    val_loader = data_loaders.load_mnist_data('val', batch_size = 128, shuffle = True) 

    # # 3. Create the DataLoaders
    # train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=16, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=16, shuffle=False)
    print("DataLoaders are ready.")

    # Training with l2-regularization
    print("\nDefining the neural network...")
    # network_MNIST = ReLUNet([784, 392, 196, 98, 49, 32, 10]) # simple MNIST network 
    network_MNIST = ReLUNet([784, 256, 128, 64, 32, 10]) # simple MNIST network 
    print("Network created.")

    # # Reload the MNIST datasets
    # mnist_train = data_loaders.load_mnist_data('train', batch_size=128, shuffle=True) # Training data
    # mnist_val = data_loaders.load_mnist_data('val', batch_size=128, shuffle=True) # Validation data 


    # Build the components of the loss function
    cross_entropy_loss = train.XEntropyReg(scalar=1.0)
    l2_loss = train.LpWeightReg(lp='l2', scalar=0.01) 

    # Build the loss function to use 
    loss_functional = train.LossFunctional(regularizers=[cross_entropy_loss, l2_loss])
    loss_functional.attach_network(network_MNIST)

    # Train the network 
    mnist_train_params = train.TrainParameters(
        # mnist_train, mnist_val, 10, loss_functional=loss_functional
        train_loader, val_loader, 10, loss_functional=loss_functional
    )
    print("\nStarting training...")
    train.training_loop(network_MNIST, mnist_train_params, use_cuda=True)

    # 6. Save the network, datasets, and dataloaders
    print("\nSaving network, datasets, and dataloaders...")

    # Define full paths for the files to be saved
    network_path = os.path.join(specific_save_dir, 'trained_network.pth')
    datasets_path = os.path.join(specific_save_dir, 'datasets.pth')
    dataloaders_path = os.path.join(specific_save_dir, 'dataloaders.pth')

    network_list = ['mnist', network_MNIST]

    # Save the trained network
    torch.save(network_list, network_path)
    # torch.save(network_MNIST, network_path)

    # Save the datasets
    datasets = {
        'train': train_loader.dataset,
        'validation': val_loader.dataset
    }
    torch.save(datasets, datasets_path)

    print("\nScript finished. All files have been saved:")
    print(f"- {network_path}")
    print(f"- {datasets_path}")
    # print(f"- {dataloaders_path}")

if __name__ == '__main__':
    main()