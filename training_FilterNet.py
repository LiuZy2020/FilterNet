import os
import glob
import numpy as np
import random
import torch
import torch.nn.functional as torchF

from DataReader_Radar import RadarDataset
from FilterNet import *
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ' '

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02) 

def gradient_penalty(critic, real, fake):
    BATCH_SIZE, c, h ,w = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,c,h,w).cuda()
    interpolated_images = real*epsilon + fake*(1-epsilon)

    #calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

def addGauseNoise(feature, SNR=0, mean=0, var=0.0001, is_Tensor = False):
    if is_Tensor:
        feature = feature.detach().numpy()
    feature_n = feature.copy()
    shape = feature_n.shape
    g_noise = np.random.normal(loc=mean, scale=var, size=shape)
    feature_n += g_noise
    feature_n = 1*(feature_n - np.min(feature_n)) / (np.max(feature_n)-np.min(feature_n))
    if is_Tensor:
        return torch.Tensor(feature_n)
    else:
        return feature_n

def show_pictures(e_model, g_model, is_DCAE=False):
    g_model.eval()
    e_model.eval()
    file_path_fall = './Data/..'
    file_path_nfall = './Data/..'

    filenames_f = sorted(glob.glob("{}/*".format(file_path_fall)))
    filenames_nf = sorted(glob.glob("{}/*".format(file_path_nfall)))
    filenames_f = np.array(filenames_f)
    filenames_nf = np.array(filenames_nf)
    np.random.shuffle(filenames_f)
    np.random.shuffle(filenames_nf)
    filenames_f = filenames_f[0:4]
    filenames_nf = filenames_nf[0:4]
    Zxx = np.zeros((8, 1, 128, 128))
    i = 0
    for ff in filenames_f:
        Zxx[i] = np.load(ff)
        Zxx[i] = Zxx[i].astype(np.float64)
        Zxx[i] = 1 * (Zxx[i] - np.min(Zxx[i])) / (np.max(Zxx[i]) - np.min(Zxx[i]))
        i += 1
    for nf in filenames_nf:
        Zxx[i] = np.load(nf)
        Zxx[i] = Zxx[i].astype(np.float64)
        Zxx[i] = 1 * (Zxx[i] - np.min(Zxx[i])) / (np.max(Zxx[i]) - np.min(Zxx[i]))
        i += 1

    batch_size = 8
    noise = np.zeros((batch_size, latent_size[0], latent_size[1], latent_size[2]))
    for i in range(batch_size):
        noise[i] = np.random.normal(loc=0, scale=var, size=noise[0].shape)
    noise = torch.Tensor(noise)

    latent = e_model(torch.tensor(Zxx.reshape(8, 1, 128, 128), dtype=torch.float).cuda())
    if is_DCAE:
        image = g_model(latent)
    else:
        image = g_model(latent, noise.cuda())

    for j in range(8):
        plt.subplot(2,4,j+1)
        #plt.colorbar()
        plt.imshow(image[j].cpu().detach().numpy().reshape(128,128), cmap='jet')
    plt.pause(0.0001)
    e_model.train()
    g_model.train()

def pretrain_FilterNet(e_model, d_model, Train_dataset, epoches=50, learningRate=0.001):
    initialize_weights(e_model)
    initialize_weights(d_model)

    E_optimizer = torch.optim.Adam(e_model.parameters(), learningRate, betas=(0.0,0.9))#weight_decay=decay)
    E_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        E_optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.000001)
    e_model.train()

    D_optimizer = torch.optim.Adam(d_model.parameters(), learningRate, betas=(0.0,0.9))#weight_decay=decay)
    D_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        D_optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.000001)
    d_model.train()

    for epoch in range(epoches):
        g_loss_total = 0
        g_loss_MSE_total = 0

        for batch in Train_dataset:
            image, label, imageN, ForN = batch  ## image.size = (64,1,128,128)
            image = image.cuda()
            imageN = imageN.cuda()
            label = label.cuda()
            ForN = ForN.cuda()
            batch_size = len(label)
            num_label = len(label[label != 0])

            latent = e_model(imageN)
            fake_image = d_model(latent)

            ## train Generator
            ### only 2-Norm
            image_diversion = (ForN - fake_image).view(batch_size,-1)
            g_loss_2Norm = torch.mean(torch.norm(image_diversion, p=2, dim=1)) \
                          /torch.mean(torch.norm(image.view(batch_size,-1), p=2, dim=1))
                        ## a trick to /||X||2, balance the learning rate of each batch

            g_loss = g_loss_2Norm
            D_optimizer.zero_grad()
            E_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            D_optimizer.step()
            E_optimizer.step()

            #### criterion ####
            g_loss_total += g_loss.item()

        D_lr_scheduler.step(g_loss_total)
        E_lr_scheduler.step(g_loss_total)
        if (epoch+1) % 10 == 0:
            print('Epoch:', epoch + 1)
            print('g_loss_total:', g_loss.item())
            #show_pictures(e_model, d_model, is_DCAE=True)
        ######

    return e_model, d_model

def validating(e_model, g_model, model, validationset):
    # This function can monitoring the performance of two test sets during training of classifier.
    with torch.no_grad():  # close grad tracking to reduce memory consumption
        e_model.eval()
        g_model.eval()
        total_correct = 0
        total_samples = 0

        for batch in validationset:
            image, labels, imageN, _ = batch
            image = image.cuda()
            labels = labels.cuda()
            imageN = imageN.cuda()
            batch_size = len(labels)
            
            latent = e_model(image)
            g_image = g_model(latent)
            model.eval()
            preds = model(g_image.detach())
            model.train()
            total_correct += preds.argmax(dim=1).eq(labels.long()).sum().item()

            total_samples += len(labels)

        val_acc = total_correct / total_samples

        return val_acc

def train_Classifier(e_model, g_model, model, Train_dataset, Validate_dataset, Test_dataset,epoches=50, learningRate=0.001, batch_size=64):
    initialize_weights(model)
    g_model.eval()
    e_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), learningRate, weight_decay=decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.000001,
    )
    model.train()

    best_test_acc = 0
    for epoch in range(epoches):
        loss_total = 0
        loss_total_ref = 0
        acc_total = 0
        total_samples = 0

        for batch in Train_dataset:
            image, label, imageN, _ = batch  ## image.size = (64,1,128,128)
            image = image.cuda()
            label = label.cuda()

            batch_size = len(label)

            latent = e_model(image)
            g_image = g_model(latent)
            preds = model(g_image.detach())
            loss = torchF.cross_entropy(preds.squeeze(1), label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            acc_total += preds.argmax(dim=1).eq(label.long()).sum().item()
            total_samples += len(label)

        print("epoch: ", epoch, "train_acc: ", acc_total / total_samples, "total_loss: ", loss_total)
        val_acc = validating(e_model, g_model, model, Validate_dataset)
        test_acc = validating(e_model, g_model, model, Test_dataset)
        print("Val_acc:", val_acc, "      Test_acc:", test_acc)
        lr_scheduler.step(loss_total)
        lr = optimizer.param_groups[0]['lr']
        print('lr: ', lr)

    return model

def train_FilterNet(e_model, g_model, d_model, Train_dataset, epoches=50, learningRate=0.001, learningRate_D=0.004):
    initialize_weights(d_model)

    E_optimizer = torch.optim.Adam(e_model.parameters(), learningRate,betas=(0.0,0.9))
    E_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        E_optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.00001)
    e_model.train()

    G_optimizer = torch.optim.Adam(g_model.parameters(), learningRate,betas=(0.0,0.9))
    G_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        G_optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.00001)
    g_model.train()

    D_optimizer = torch.optim.Adam(d_model.parameters(), learningRate_D,betas=(0.0,0.9))
    D_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        D_optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=0.00001)
    d_model.train()


    stop_point = 100
    for epoch in range(epoches):
        g_loss_total = 0
        d_loss_total = 0
        g_loss_MSE_total = 0

        for batch in Train_dataset:
            image, label, imageN, ForN = batch  ## image.size = (64,1,128,128)
            image = image.cuda()
            imageN = imageN.cuda()
            label = label.cuda()
            ForN = ForN.cuda()
            batch_size = len(label)
            num_label = len(label[label != 0])

            ## train Discriminator
            for _ in range(1):
                ForN_fall = ForN[torch.sum(ForN, dim=(1, 2, 3)) > 0]
                d_real_result = d_model(ForN_fall)
                latent = e_model(imageN)
                fake_image = g_model(latent)
                fake_image_fall = label.reshape(batch_size, 1, 1, 1).repeat(1,1,128,128) * fake_image
                fake_image_fall = fake_image_fall[torch.sum(fake_image_fall, dim=(1, 2, 3)) != 0]

                d_fake_result = d_model(fake_image_fall.detach()) 
                gp = gradient_penalty(d_model, ForN_fall, fake_image_fall)
                d_loss = -(torch.mean(d_real_result)-torch.mean(d_fake_result)) + LAMBDA_GP*gp
                D_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                D_optimizer.step()

            ## train Generator
            g_result = d_model(fake_image_fall)
            ### GAN loss + 2-Norm
            image_diversion = (ForN - fake_image).view(batch_size,-1)
            g_loss_2Norm = torch.mean(torch.norm(image_diversion, p=2, dim=1)/torch.norm(image.view(batch_size,-1), p=2, dim=1)) #\still a trick
            g_loss_GAN = -torch.mean(g_result)
            g_loss = Alpha*g_loss_2Norm + g_loss_GAN
            ####
            G_optimizer.zero_grad()
            E_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            G_optimizer.step()
            E_optimizer.step()

            #### criterion ####
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            g_loss_2Norm_total += g_loss_2Norm.item()
            print('g_loss_GAN:', g_loss_GAN.item(), '   g_loss_2Norm:', g_loss_2Norm.item())

        G_lr_scheduler.step(g_loss_2Norm_total/ (1920//64))
        E_lr_scheduler.step(g_loss_2Norm_total/ (1920//64))
        D_lr_scheduler.step(d_loss_total)
        print('Epoch:', epoch + 1)
        print('---------g_loss_2Norm_mean:', (g_loss_2Norm_total / (1920 // 64)), '------------')
        if (epoch+1) % 10 == 0:
            print('d_loss:', d_loss_total, 'g_loss:', g_loss_total)
            #show_pictures(e_model, g_model, is_DCAE=True)

    return e_model, g_model, d_model

decay = 4e-5
LAMBDA_GP = 12
lr_r = 0.004
lr_d = 0.004
Alpha = 12 # this param should be adjusted on different computer perhaps
index = '2'
latent_size = [8,8,8]
var = 0.001
pre_epoches = 100
gan_epoches = 100

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    seed_torch(66)
    print(torch.__version__)
    file_path = './Data/Trainset/'
    Dataset_ = RadarDataset(file_path,len(file_path))
    batch_size = 64
    train_dataset = torch.utils.data.DataLoader(Dataset_, batch_size=batch_size, shuffle=True)
    file_path_test = './Data/..'
    Dataset_V = RadarDataset(file_path_test, len(file_path_test))
    validation_dataset = torch.utils.data.DataLoader(Dataset_V, batch_size=batch_size, shuffle=True)
    file_path_test = './Data/..'
    Dataset_T = RadarDataset(file_path_test, len(file_path_test))
    test_dataset = torch.utils.data.DataLoader(Dataset_T, batch_size=batch_size, shuffle=True)
    ###start training
    E = encoder(input_channels=1).cuda()
    De = decoder(input_channels=1).cuda()
    print('--------------Pre-train the encoder------------------ ')
    e_model, de_model = pretrain_FilterNet(E, De, train_dataset, epoches=pre_epoches, learningRate=0.001) #50 epoches
    torch.save(e_model,'./Model_saved/model_encoder_pre_'+index+'.pth')
    torch.save(de_model, './Model_saved/model_decoder_'+index+'.pth')
    print('----------------end pre-train------------------------- ')
    e_model = torch.load('./Model_saved/model_encoder_pre_'+index+'.pth').cuda()
    g_model = torch.load('./Model_saved/model_decoder_'+index+'.pth').cuda()
    d_model = Discriminator(input_channels=1, num_classes=1).cuda()
    e_model, g_model, d_model = train_FilterNet(e_model, g_model, d_model, train_dataset,
                                                epoches=gan_epoches, learningRate=lr_r, learningRate_D=lr_d)
    torch.save(g_model,'./Model_saved/model_generator_'+index+'.pth')
    torch.save(d_model, './Model_saved/model_discriminator_'+index+'.pth')
    torch.save(e_model, './Model_saved/model_encoder_'+index+'.pth')
    print('-------------end generator training----------------')
    print('-------------classifier training------------------- ')
    C = simple_Classifier(input_channels=1, num_classes=2).cuda()
    c_model = train_Classifier(e_model,g_model, C, train_dataset, validation_dataset, test_dataset,epoches=50, learningRate=0.001)
    torch.save(c_model, './Model_saved/model_classifier_'+index+'.pth')
    validating(e_model, g_model,c_model, test_dataset)


if __name__ == '__main__':
    main()
