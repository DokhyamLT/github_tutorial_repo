import numpy as np
from model_arch import _mouth_G, _mouth_F, _discriminator, _discriminatorF, _discriminatorS
import torch
import torch.nn as nn
import cv2
import get_inputs_makeup
import os
from torch.utils import data
import torch.optim as optim
import dlib
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp, estimate_transform


def printCUDA(x):
    print(x.cpu().data.numpy().shape)

def get_lms(im):

	rect = dlib.rectangle(0,0,im.shape[0],im.shape[1])
	predictor = dlib.shape_predictor('/Users/dokhyam/Downloads/shape_predictor_68_face_landmarks.dat')
	lms_face_im = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),rect)
	PoI = [8,31,35,48,54]
	my_points = np.zeros((len(PoI),2))
	for i in range(len(PoI)):
		my_points[i,:] = [int(lms_face_im.part(PoI[i]).x),int(lms_face_im.part(PoI[i]).y)]
	my_points = np.array(my_points, dtype = np.int)
	top_nose = max(my_points[1,1],my_points[2,1])
	return [int(top_nose),my_points[0,1],my_points[3,0],my_points[4,0]]

def get_lms_all(im):
	rect = dlib.rectangle(0,0,im.shape[0],im.shape[1])
	predictor = dlib.shape_predictor('/Users/dokhyam/Downloads/shape_predictor_68_face_landmarks.dat')
	lms_face_im = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),rect)
	points = np.zeros((68,2))
	for i in range(68):
		points[i,:] = (lms_face_im.part(i).x,lms_face_im.part(i).y)
	return points

def np_im(im_tensor):
	imt = im_tensor.numpy().transpose(1,2,0)
	return np.array(imt, dtype=np.uint8)


def add_boundaries(pts, im1):
	new_pts = np.vstack([pts, 
	[im1.shape[1]-1,0], 
	[im1.shape[1]-1,im1.shape[0]-1],
	[0, im1.shape[0]-1],
	[0,0],
	[int(im1.shape[1]/2),0],
	[0,int(im1.shape[0]/2)],	
	[im1.shape[1]-1,int(im1.shape[0]/2)]
	])
	return new_pts

def warp_2_ims(im_no_mu,im_mu):
	pts1 = get_lms_all(im_no_mu)
	pts2 = get_lms_all(im_mu)
	pts_av = (pts1 + pts2)/2
	pts1 = add_boundaries(pts1, im_no_mu)
	pts2 = add_boundaries(pts2, im_mu)
	pts_av = add_boundaries(pts_av, im_mu)
	M1 = estimate_transform('piecewise-affine',pts2,pts_av)
	warped_im1 = warp(im_mu, M1.inverse,output_shape=im_size, mode='edge')
	M2 = estimate_transform('piecewise-affine',pts1,pts_av)
	warped_im2 = warp(im_no_mu, M2.inverse,output_shape=im_size, mode='edge')
	added_ims = np.array((warped_im1/2 + warped_im2/2)*255, dtype = np.uint8)
	all_ims = np.hstack((im_no_mu,im_mu,added_ims))
	return added_ims

def float_tensor(tensor):
	if device == 'cuda':
		tensor = tensor.type('torch.cuda.FloatTensor')
	else:
		tensor= tensor.type('torch.FloatTensor')
	return tensor

# -----------------------------
#.         PARAMETERS
# -----------------------------
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
batchSize =4
ngf = 64
ndf = 64
nc = 3
lrt = 0.001
lambda_G = 0.1
lambda_F = 0.1
lambda_p = 0.1
folder_path = '/Users/dokhyam/Documents/makeup_project/cycle_dataset1/'
mu_data_path_train = 'trainA/'
no_mu_data_path_train = 'trainB/'
mu_data_path_test = 'testA/'
no_mu_data_path_test = 'testB/'
im_size = (256, 256)
max_epochs = 1
# ------------------------------
#         PREPARE DATA
# ------------------------------
partition = {}
partition['train'] = {}
partition['test'] = {}
partition['train']['makeup'] = os.listdir(folder_path + mu_data_path_train)
partition['train']['no_makeup'] = os.listdir(folder_path + no_mu_data_path_train)
partition['test']['no_makeup'] = os.listdir(folder_path + no_mu_data_path_test)
partition['test']['makeup'] = os.listdir(folder_path + mu_data_path_test)

mu_sampler = data.BatchSampler(data.RandomSampler(partition['train']['makeup']),batch_size = batchSize, drop_last = True)
no_mu_sampler = data.BatchSampler(data.RandomSampler(partition['train']['no_makeup']),batch_size = batchSize, drop_last = True)
data_loader_params_ims = {'batch_sampler': mu_sampler,
						'num_workers': 12}
data_loader_params_masks = {'batch_sampler': mu_sampler,
							'num_workers': 12}

class Dataset(data.Dataset):
  def __init__(self, list_IDs,is_im, labels = None):
        self.list_IDs = list_IDs
        self.is_im = is_im
        self.max_height = 0
        self.max_width = 0

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
    	ID = self.list_IDs[index]
    	if self.is_im:
    		X = cv2.imread(folder_path + mu_data_path_train+ ID)
    	else:
    		X = cv2.imread(folder_path + no_mu_data_path_train+ ID)
    	X = cv2.resize(X,im_size)
    	tblr = get_lms(X)
    	X = np.transpose(X,(2,0,1))

    	return  X, tblr


training_set_ims = Dataset(partition['train']['makeup'], True)
training_generator_ims = data.DataLoader(training_set_ims, **data_loader_params_ims)
training_set_masks = Dataset(partition['train']['no_makeup'], False)
training_generator_masks = data.DataLoader(training_set_masks, **data_loader_params_masks)

# --------------------------
#          MODELS
# --------------------------
mouth_G = _mouth_G()
mouth_F = _mouth_F()
discriminator = _discriminator(batch_size=batchSize)
discriminatorF = _discriminatorF(batch_size=batchSize)
discriminatorS = _discriminatorS(batch_size=batchSize)
print(discriminator)

# im1_mouth,im1_all = get_inputs_makeup.main(0)
# im2 = np.load('/Users/dokhyam/Documents/makeup_project/cropped_mouth/024_1.npy')
# im1 = cv2.resize(im1_mouth,(60,50))
# im2 = cv2.resize(im2,(60,50))
# im1_all = cv2.resize(im1_all,(256,256))
# im1_all = im1_all[np.newaxis,...].transpose((0,3,1,2))
# im1_all = torch.from_numpy(im1_all).type('torch.FloatTensor')
mouth_G = mouth_G.to(device)
mouth_F = mouth_F.to(device)
discriminator = discriminator.to(device)
discriminatorF = discriminatorF.to(device)
discriminatorS = discriminatorS.to(device)
criterionL1 = nn.L1Loss()
criterionCE = nn.BCELoss()
criterionCE = criterionCE.to(device)
criterionL1 = criterionL1.to(device)
for epoch in range(max_epochs):
	optimizerG = optim.Adam(mouth_G.parameters(), lr=lrt*100, betas=(0.9, 0.999))
	optimizerF = optim.Adam(mouth_F.parameters(), lr=lrt, betas=(0.9, 0.999))
	optimizerDG = optim.Adam(discriminator.parameters(), lr=lrt, betas=(0.9, 0.999))
	optimizerDF = optim.Adam(discriminatorF.parameters(), lr=lrt, betas=(0.9, 0.999))
	optimizerDP = optim.Adam(discriminatorS.parameters(), lr=lrt, betas=(0.9, 0.999))
	for (mu_im,tblr),(no_mu_im,tblr2) in zip(training_generator_ims, training_generator_masks):
		warped_ims = torch.ones((batchSize,nc, im_size[0],im_size[1]))
		for i in range(batchSize):
			warped_ims[i,...] = torch.from_numpy(np.transpose(warp_2_ims(np_im(no_mu_im[i]),np_im(mu_im[i])), (2,0,1)))
		if device == 'cuda':
			mu_im, no_mu_im, warped_ims = mu_im.type('torch.cuda.FloatTensor'), no_mu_im.type('torch.cuda.FloatTensor'),warped_ims.type('torch.cuda.FloatTensor')
		else:
			mu_im, no_mu_im, warped_ims = mu_im.type('torch.FloatTensor'), no_mu_im.type('torch.FloatTensor'),warped_ims.type('torch.FloatTensor')

		labs_GAN_pos = torch.ones((batchSize,1,30,30)).to(device)
		labs_GAN_neg = torch.zeros((batchSize,1,30,30)).to(device)		
		# ------------------------------
		#    TRAIN DISCRIMINATOR for G
		# ------------------------------
		optimizerDF.zero_grad()
		optimizerDG.zero_grad()
		optimizerG.zero_grad()
		optimizerF.zero_grad()
		optimizerDP.zero_grad()
		full_pred = no_mu_im
		mu_crop = torch.zeros((batchSize,nc,90,135))
		no_mu_crop = torch.zeros((batchSize,nc,90,135))

		for i in range(batchSize):
			# full_pred[i,...] = cut_and_add(i,mu_im,no_mu_im, tblr, tblr2)
			
			mu_crop[i,:,:(min((tblr[0][i].item()+90),256)- tblr[0][i].item()),:(min((tblr[2][i].item()+135),256)- tblr[2][i].item())] = mu_im[i,:,tblr[0][i]:(tblr[0][i]+90),tblr[2][i]:(tblr[2][i]+135)]
			no_mu_crop[i,:,:(min((tblr2[0][i].item()+90),256)- tblr2[0][i].item()),:(min((tblr2[2][i].item()+135),256)- tblr2[2][i].item())] = no_mu_im[i:i+1,:,tblr2[0][i]:(tblr2[0][i]+90),tblr2[2][i]:(tblr2[2][i]+135)]
		mu_crop = mu_crop.to(device)
		no_mu_crop = no_mu_crop.to(device)
		pred_ims = mouth_G(no_mu_crop, mu_crop)
		full_pred = float_tensor(full_pred)
		for i in range(batchSize):

			full_pred[i,:,tblr2[0][i]:((tblr2[0][i]+(min((tblr2[0][i].item()+89),256)))- tblr2[0][i].item()),tblr2[2][i]:((tblr2[2][i]+(min((tblr2[2][i].item()+134),256)))-tblr2[2][i].item())] = full_pred[i,:,tblr2[0][i]:((tblr2[0][i]+(min((tblr2[0][i].item()+89),256)))- tblr2[0][i].item()),tblr2[2][i]:((tblr2[2][i]+(min((tblr2[2][i].item()+134),256)))-tblr2[2][i].item())].clone() + pred_ims[i,:,:(min((tblr2[0][i].item()+89),256)- tblr2[0][i].item()),:(min((tblr2[2][i].item()+134),256)- tblr2[2][i].item())].clone().to(device)
		dis_inp = full_pred
		dis_pred_G = discriminator(dis_inp)
		dis_real_mu = discriminator(mu_im)
		fake_loss = criterionCE(dis_pred_G, labs_GAN_neg)
		real_loss = criterionCE(dis_real_mu, labs_GAN_pos)
		discriminatorG_loss = fake_loss + real_loss
		discriminatorG_loss.backward(retain_graph=True)
		optimizerDG.step()
		# ------------------------------
		#    TRAIN DISCRIMINATOR for F
		# ------------------------------
		full_pred_mu = mu_im
		F_pred = mouth_F(mu_crop)
		for i in range(batchSize):
			full_pred_mu[i,:,tblr[0][i]:((tblr[0][i]+(min((tblr[0][i].item()+89),256)))- tblr[0][i].item()),tblr[2][i]:((tblr[2][i]+(min((tblr[2][i].item()+134),256)))-tblr[2][i].item())] = full_pred_mu[i,:,tblr[0][i]:((tblr[0][i]+(min((tblr[0][i].item()+89),256)))- tblr[0][i].item()),tblr[2][i]:((tblr[2][i]+(min((tblr[2][i].item()+134),256)))-tblr[2][i].item())].clone() + pred_ims[i,:,:(min((tblr[0][i].item()+89),256)- tblr[0][i].item()),:(min((tblr[2][i].item()+134),256)- tblr[2][i].item())].clone().to(device)
		F_dis_ins = full_pred_mu
		dis_pred_F = discriminator(F_dis_ins)
		dis_real_no_mu = discriminator(no_mu_im)
		fake_loss = criterionCE(dis_pred_F, labs_GAN_neg)
		real_loss = criterionCE(dis_real_no_mu, labs_GAN_pos)
		discriminatorF_loss = fake_loss + real_loss
		discriminatorF_loss.backward(retain_graph=True)
		optimizerDF.step()
		# ------------------------------
		#     LOSS GAN for G
		# -------------------------------
		# G_pred = mouth_G(no_mu_im, mu_im)
		# dis_pred_G = discriminator(G_pred)
		loss_GAN_G = criterionCE(dis_pred_G,labs_GAN_pos)
		# ------------------------------
		#     LOSS GAN for F
		# ------------------------------- 
		# F_pred = mouth_F(mu_im)
		# dis_pred_F = discriminatorF(F_pred)
		loss_GAN_F = criterionCE(dis_pred_F,labs_GAN_neg) 
		# ------------------------------
		#     IDENTITY LOSS
		# -------------------------------    
		# x_pred = mouth_F(mouth_G(no_mu_im, mu_im))
		# loss_i = criterionL1(x_pred, no_mu_im)
		x_pred = mouth_F(mouth_G(no_mu_crop, mu_crop))
		loss_i = criterionL1(x_pred, no_mu_crop)
		# ------------------------------
		#     STYLE LOSS
		# ------------------------------
		# mu_style_pred = mouth_G(mouth_F(mu_im),mouth_G(no_mu_im,mu_im))
		# loss_s = criterionL1(mu_style_pred, mu_im)
		mu_style_pred = mouth_G(mouth_F(mu_crop),mouth_G(no_mu_crop,mu_crop))
		loss_s = criterionL1(mu_style_pred, mu_crop)
		# ------------------------------
		# ------------------------------
		#     BLEND LOSS
		# ------------------------------	
		pred_blend = discriminatorS(warped_ims)
		loss_p = criterionCE(pred_blend,labs_GAN_pos) + criterionCE(dis_pred_G, labs_GAN_neg)

		#-------------------------------	
		loss_all = lambda_G*loss_GAN_G + lambda_F*loss_GAN_F + loss_i +  loss_s #+ lambda_p*loss_p
		loss_all.backward()
		optimizerG.step()
		optimizerF.step()
		optimizerDP.step()
		print("epoch: " + str(epoch) + " batch: " + str(counter_batch) + "loss_all: " + str(loss_all.item()) + " loss discriminator: " + str(discriminatorG_loss.item()))
		