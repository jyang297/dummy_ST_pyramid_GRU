import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.RIFE import Model
import argparse
from model.pretrained_RIFE_loader import IFNet_update
from model.pretrained_RIFE_loader import convert_load

# Config
root = '/root/MyCode/Valid/dummy_ST_pyramid_GRU'
output_root = "/root/autodl-tmp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_path = "/root/autodl-fs/Origin/dancingGirls"
# frame_path = output_root + "/outputs/soccer_4x" 
# frame_path = "outputs/outputs_slowmotion"                

output_path = output_root + "/outputs/dancing/dancingGirls_2x"
pretrained_model_path = root + '/intrain_log'
pretrained_path = root + '/RIFE_log' # pretrained RIFE path
shift = 0



def load_frames(frame_folder, start_frame, num_frames=4):
    # Load a sequence of 'num_frames' starting from 'start_frame'
    frames = []
    for i in range(1,num_frames+1):
        frame_path = os.path.join(frame_folder, f"frame_{start_frame + i:04d}.png")
        frame = Image.open(frame_path).convert('RGB')
        frames.append(frame)
        if i != num_frames:
            frames.append(frame)
        
    # print('load')
    return frames




def preprocess_frames(frames):
    # Convert frames to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # tensor = torch.stack([transform(frame) for frame in frames], dim=1)  # shape: (3, 7, H, W)
    tensor = torch.stack([transform(frame) for frame in frames], dim=0) # shape: (3, 7, H, W)
    #print("Tensor.shape", tensor.shape)
    tensor = tensor.view(1, 3*7, tensor.shape[2], tensor.shape[3])  # shape: (1, 21, H, W)
    
    #print('preprocess')
    return tensor.to(device)

def save_frame(tensor, output_folder, frame_index):
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    img.save(os.path.join(output_folder, f"frame_{frame_index:04d}.png"))
    ##print('save\n')

def inference_video(model, frame_folder, output_folder, total_frames):
    with torch.no_grad():
        for start_frame in range(0,total_frames - 2 + 1, 3):  # Adjust the step to handle overlap or gaps

            # manual shift
            start_frame += shift
            i = int(start_frame/3)
            frames = load_frames(frame_folder, start_frame)
            save_start_point = i*6
            input_tensor = preprocess_frames(frames)
            #print('gointo model')
            output_allframes_tensors = model(input_tensor)
            #print('compute finished')
            interpolated_frames = output_allframes_tensors[:-1] 
            # Example reshape to (1, 7, 3, H, W)
            print('try save')
            # Saving only interpolated frames: indices 0 to 5. the 6th is the 0th of the next saving loop
            for i in range(6):
                save_frame(interpolated_frames[i, :, :, :], output_folder, save_start_point + i + 1)
            torch.cuda.empty_cache()





if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=7, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()

 
    torch.cuda.set_device(args.local_rank)

    torch.backends.cudnn.benchmark = True
    pretrained_path = '/root/MyCode/Valid/dummy_ST_pyramid_GRU/RIFE_log'
    checkpoint = convert_load(torch.load(f'{pretrained_path}/flownet.pkl', map_location=device))
    Ori_IFNet_loaded = IFNet_update()
    Ori_IFNet_loaded.load_state_dict(checkpoint)
    for param in Ori_IFNet_loaded.parameters():
        param.requires_grad = False

    model = Model(Ori_IFNet_loaded, args.local_rank)
    pretrained_model_path = '/root/MyCode/Valid/dummy_ST_pyramid_GRU/intrain_log'
    model.load_model(pretrained_model_path)
    print("Loaded ConvLSTM model")
    model.eval()


    inference_video(model.simple_inference, frame_path, output_path, 2000)
