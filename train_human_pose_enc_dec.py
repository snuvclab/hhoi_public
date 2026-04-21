import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from datasets import Dataset
import pickle
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
import os
import math
import numpy as np
from glob import glob
import cv2
from pytorch3d.transforms import *

# Define the Encoder-Decoder Model
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(126, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Tanh()  # Tanh activation to ensure output is in range [-1, 1]
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 126),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Pass through the encoder
        encoded = self.encoder(x)
        # Pass through the decoder
        decoded = self.decoder(encoded)
        return encoded, decoded
    
@dataclass
class TrainingConfig_HumanPose:
    train_batch_size = 4096#65536#1048576
    eval_batch_size = 64  # how many images to sample during evaluation
    num_epochs = 100
    learning_rate = 1e-3
    lr_warmup_steps = 1000
    output_dir = "enc_dec_human_pose"

def save_model(model, path='model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    
def load_model(model, path='model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    return model

def infer(model, input_data):
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for inference
        encoded, decoded = model(input_data)
    return encoded, decoded
    
def train_model(model, dataloader, criterion, optimizer, lr_scheduler, config, device):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False)
        
        for batch in progress_bar:
            human_poses = batch["human_poses"].to(device)
            encoded, decoded = model(human_poses)
        
            loss = criterion(decoded, human_poses)
            
            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters
            
            # Update the learning rate
            lr_scheduler.step()

            # Accumulate the loss for logging
            batch_loss = loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix({"Batch Loss": batch_loss})
        
        # Log the epoch loss
        total_loss += epoch_loss
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.6f}")
        
        # save model
        if epoch % 1 == 0:
            os.makedirs(config.output_dir, exist_ok=True)
            save_model(model, f'{config.output_dir}/encoder_decoder_epoch_{epoch}.pth')
            
    os.makedirs(config.output_dir, exist_ok=True)
    save_model(model, f'{config.output_dir}/encoder_decoder_latest.pth')
    
    return model


if __name__ == "__main__":
    # Load masks, camera intrinsics, and initialize 3D points here...
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', help='train | infer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train':
        config = TrainingConfig_HumanPose()
        with open('data/human_pose/human_pose.pkl', 'rb') as f:
            training_data = pickle.load(f)
            
        
        dataset = Dataset.from_dict(training_data)

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
        dataset.set_format("torch", columns=["human_poses"])

        # Create the model
        model = EncoderDecoder().to(device)

        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )

        trained_model = train_model(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            device=device
        )
    else:                
        human_pose_data = []
        for scenario in os.listdir(f"data/hoi"):
            for next_pkl in tqdm(sorted(glob(f"data/hoi/{scenario}" + "/*.pkl"))):
                with open(next_pkl, "rb") as f:
                    pkl_content = pickle.load(f)

                
        for scenario in os.listdir(f"data/hhi"):
            for next_pkl in tqdm(sorted(glob(f"data/hhi/{scenario}" + "/*.pkl"))):
                with open(next_pkl, "rb") as f:
                    pkl_content = pickle.load(f)

                left_body_pose = pkl_content['left']['body_pose']
                right_body_pose = pkl_content['right']['body_pose']
                human_pose_data.append(left_body_pose)
                human_pose_data.append(right_body_pose)

        with open('data/human_pose/human_pose.pkl', 'rb') as f:
            training_data = pickle.load(f)

        human_pose_data = np.concatenate(human_pose_data, axis=0)
        
        sixd_list = []
        for i in range(21):
            next_theta = human_pose_data[..., 3 * i : 3 * i + 3]
            next_rot_mat = torch.tensor(np.array([cv2.Rodrigues(row.reshape(3, 1))[0] for row in next_theta]))
            next_6d = matrix_to_rotation_6d(next_rot_mat)
            sixd_list.append(next_6d)
        base_theta = torch.cat(sixd_list, dim=-1).to(device)
        
        all_theta = training_data['human_poses'].to(device)
        
        config = TrainingConfig_HumanPose()
        import matplotlib.pyplot as plt
        epoch_list = []
        mse_list = []
        
        for e in tqdm(range(100)):
            ckpt = f"{config.output_dir}/encoder_decoder_epoch_{e}.pth"
            if not os.path.exists(ckpt):
                print(f"[skip] missing: {ckpt}")
                continue

            model = EncoderDecoder().to(device)
            model = load_model(model, ckpt)
            model.eval()

            with torch.no_grad():
                _, decoded_base = infer(model, base_theta)
                mse = F.mse_loss(base_theta, decoded_base).item()

            epoch_list.append(e)
            mse_list.append(mse)

        plt.figure()
        plt.plot(epoch_list, mse_list, label="base_theta")
        plt.xlabel("Checkpoint epoch")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("mse_vs_epoch.png", dpi=200)
        plt.close()
        
        import pdb
        pdb.set_trace()
        
        dec_pose_list = []
        for i in range(21):
            nx_pose = decoded_output[:, 6 * i : 6 * i + 6]
            nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
            nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
            dec_pose_list.append(nx_mats)
        dec_pose = np.concatenate(dec_pose_list, axis=-1)
        
        base_pose_list = []
        for i in range(21):
            nx_pose = base_theta[:, 6 * i : 6 * i + 6]
            nx_mat = rotation_6d_to_matrix(nx_pose.cpu()).numpy()
            nx_mats = np.array([cv2.Rodrigues(R)[0].flatten() for R in nx_mat])
            base_pose_list.append(nx_mats)
        base_pose = np.concatenate(base_pose_list, axis=-1)
        import pdb
        pdb.set_trace()

        if args.feat_type == 'rot':
            R_list = []
            samples = np.random.uniform(low=0, high=360, size=(10, 3))
            for i in range(samples.shape[0]):
                r, p, y = samples[i]
                R = make_rotate(math.radians(r), math.radians(p), math.radians(y))
                R_list.append(R)
            R_np = np.stack(R_list)
            input_data = torch.tensor(R_np, dtype=torch.float32).reshape(-1, 9).to(device)
            encoded_output, decoded_output = infer(model, input_data)
            rot_est = robust_compute_rotation_matrix_from_ortho6d(decoded_output)
        elif args.feat_type == 'trans':
            samples = (torch.rand(10, 3) * 10) - 5
            input_data = samples.to(device)
            encoded_output, decoded_output = infer(model, input_data)
        elif args.feat_type == 'scale':
            samples = (torch.rand(10, 3) * (10 - 0.01)) + 0.01
            input_data = samples.to(device)
            encoded_output, decoded_output = infer(model, input_data)
            
        # import pdb
        # pdb.set_trace()

        print("Encoded output (3D vectors in [-1, 1]):")
        print(encoded_output)
        print("\nDecoded output (reconstructed 6D vectors):")
        print(decoded_output)