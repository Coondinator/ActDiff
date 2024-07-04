import os
import wandb
import numpy as np
import torch
import math
import time

from tqdm.auto import tqdm
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import dataset, Dataset, DataLoader

from dataset.dataset import Action_Dataset
from argument import Arguments
from model.diffusion import ActDiff

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(40)
torch.cuda.manual_seed(20)
np.random.seed(10)


def basic_train(model,
                train_dataloader,
                test_dataloader,
                n_epoch,
                save_path,
                loss_type="l2",
                lrate=1e-3,
                train_mode='noise'):
    output_path = os.path.join(save_path, time.strftime("%Y-%m-%d-%H"))
    os.makedirs(output_path, exist_ok=True)

    output_model_path = os.path.join(output_path, 'model')
    os.makedirs(output_model_path, exist_ok=True)

    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    model.train()
    wandb.init(
        # set the wandb project where this run will be logged
        project="LDM_Sequence_Transformer",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lrate,
            "architecture": "ldm",
            "dataset": "MotionX",
            "epoch": 1000,
        }
    )

    for ep in range(n_epoch + 1):
        print(f'epoch {ep}')
        start_time = time.time()
        acc_train_loss = 0
        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        # pbar = tqdm(train_dataloader)
        for index, (action, cond, label) in enumerate(train_dataloader):  # x: images
            # latent, text = x[0], x[1]
            optim.zero_grad()
            action = action.to(device)  # [batch, action_length, 16]
            cond = cond.to(device)  # [batch, cond_length, 16]
            label = label.to(device)  # [batch, 1]
            # perturb data

            noise = torch.randn_like(action)
            t = torch.randint(0, args.timesteps, (action.shape[0],)).to(device)
            loss = model(x=action, cond_act=cond, noise=noise, label=label, t=t)
            acc_train_loss += loss.item()
            loss.backward()

            optim.step()

        acc_val_loss = 0
        with torch.no_grad():
            model.eval()
            for test_action, test_cond, test_label in test_dataloader:
                test_action = test_action.to(device)
                test_cond = test_cond.to(device)
                test_label = test_label.to(device)  # [batch, 1]

                # attention_mask = generate_attention_mask(test_data)
                noise = torch.randn_like(test_action)
                t = torch.randint(0, args.timesteps, (test_action.shape[0],)).to(device)
                val_loss = model(x=test_action, noise=noise, cond_act=test_cond, label=test_label, t=t)
                acc_val_loss += val_loss.item()
                # print('test_loss:', test_loss.item())

        acc_train_loss /= len(train_dataloader)
        acc_val_loss /= len(test_dataloader)
        end_time = time.time()
        epoch_time = end_time - start_time

        if (ep % 200) == 0:
            torch.save(model.state_dict(), os.path.join(output_model_path, f'ActDiff_xstart_{ep}.pt'))


        wandb.log({"acc_train_loss": acc_train_loss, "test_loss": acc_val_loss, "epoch_time": epoch_time})
        print('[{:03d}/{}] acc_train_loss: {:.4f}\t test_loss: {:.4f}'.format(
            ep, n_epoch, acc_train_loss, acc_val_loss))

    torch.save(model.state_dict(), os.path.join(output_model_path, f'ActDiff_xstart_final.pt'))
    wandb.finish()

    # basic_eval(model=ldm_model, nemf_model=nemf_model, fk=fk, prompt=, output_path=output_path)
    return


def basic_eval(model,  cond, output_path):
    output_eval_path = os.path.join(output_path, 'eval')
    os.makedirs(output_eval_path, exist_ok=True)

    diffusion_model = model
    result = torch.rand([1, 5, 1280]).to(device)

    sample, _ = diffusion_model.p_sample_loop_x_start(cond, result.shape)

    return


def train():
    data_path = './data/try/pant_blackjean/'
    dataset = Action_Dataset(dataset_path=data_path, args=args)
    batch_size = 512
    lr = 1e-4

    act_diff = ActDiff(args).to(device)
    dataset_size = len(dataset)

    train_dataset_size = int(0.8 * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size])
    # visual_set, _ = torch.utils.data.random_split(test_set, [3, len(test_set) - 3])

    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    basic_train(model=act_diff,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                n_epoch=800,
                save_path='./checkpoint/',
                lrate=lr,
                train_mode='x_start')



if __name__ == '__main__':

    from argparse import ArgumentParser
    configure_path = 'model.yaml'

    parser = ArgumentParser()
    args = Arguments('./model', filename=configure_path)
    train()
    # test_save_video()
