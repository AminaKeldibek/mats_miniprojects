import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# A simple Setup/Cleanup for the "Simulation"
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' so this runs on your Mac CPU or standard Linux CPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_and_save(rank, world_size):
    # 1. Start the simulation
    setup(rank, world_size)
    
    # 2. Create Model
    # We use a fixed seed so all models start with same weights (important!)
    torch.manual_seed(42) 
    model = nn.Linear(10, 1).to("cpu")
    
    # 3. Wrap in DDP
    ddp_model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 4. Training Step (Simulated)
    # Forward
    outputs = ddp_model(torch.randn(20, 10))
    loss = outputs.sum()
    
    # Backward (The All-Reduce Sync happens here!)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # --- MODEL SAVING SECTION ---
    
    # We use a barrier to make sure all processes have finished training
    # before we think about saving.
    dist.barrier()
    
    if rank == 0:
        print(f"Rank {rank}: Everyone finished. I am saving the model now...")
        
        # Define where to save
        save_path = "my_distributed_model.pt"
        
        # CRITICAL: Access .module to get the real model inside the DDP wrapper
        # If we just saved ddp_model.state_dict(), the keys would all start with 'module.'
        state_dict = ddp_model.module.state_dict()
        
        torch.save(state_dict, save_path)
        
        print(f"✅ Model saved to: {os.getcwd()}/{save_path}")
        print("   (You can load this file normally without DDP now!)")
        
    # Wait for Rank 0 to finish saving before exiting
    dist.barrier()
    
    cleanup()

def verify_saved_model():
    """
    This runs completely separately, acting like a user 
    loading the model after training is done.
    """
    print("\n--- Verification Phase ---")
    try:
        # Load the file normally
        state_dict = torch.load("my_distributed_model.pt")
        
        # Create a fresh model (not DDP)
        new_model = nn.Linear(10, 1)
        new_model.load_state_dict(state_dict)
        
        print("🎉 Success! The saved model loaded perfectly into a standard nn.Linear.")
        print(f"Weights: {new_model.weight.data}")
        
        # Clean up the file
        os.remove("my_distributed_model.pt")
        
    except FileNotFoundError:
        print("❌ Error: The model file was not found. Did training fail?")

if __name__ == "__main__":
    world_size = 4
    print(f"🚀 Spawning {world_size} processes...")
    
    mp.spawn(train_and_save,
             args=(world_size,),
             nprocs=world_size,
             join=True)
             
    # After the simulation finishes, we verify the file exists and works
    verify_saved_model()