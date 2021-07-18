import numpy as np
import os
import datetime
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Add
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense, Lambda
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import backend as K
from math import ceil
import matplotlib.pyplot as plt
from data_loader import DataLoader


class SRGAN():
	def __init__(self, lr_height=64, lr_width=64, channels=3, upscaling_factor=4, training=True):
		"""
		:param int lr_height: height of low resolution images
		:param int lr_width: width of low resolution images
		:param int channels: image channels
		:param int upscaling_factor: image upscaling factor
		:param bool training: if model is in training mode
		"""

		# Low-resolution image dimensions
		self.lr_height = lr_height
		self.lr_width = lr_width

		# High-resolution image dimensions
		if upscaling_factor % 2 != 0:
			raise ValueError('Upscaling factor must be a multiple of 2')
		self.upscaling_factor = upscaling_factor
		self.hr_height = int(self.lr_height * self.upscaling_factor)
		self.hr_width = int(self.lr_width * self.upscaling_factor)

		# Low- and high-resolution shapes
		self.channels = channels
		self.lr_shape = (self.lr_height, self.lr_width, self.channels)
		self.hr_shape = (self.hr_height, self.hr_width, self.channels)

		# Training mode
		self.training = training

		# Number of filters in the first layer of generator and discriminator 
		self.g_filters = 64
		self.d_filters = 64

		# Networks learning rates
		self.vgg_lr = 1e-4
		self.gen_lr = 1e-4
		self.dis_lr = 1e-4

		# TODO: read and pick the best loss functions for networks and loss weights
		# Networks loss functions
		self.vgg_loss = 'mse'
		self.gen_loss = 'mse'
		self.dis_loss = 'binary_crossentropy'
		self.loss_weights = [1e-3, 1e-2]

		# Build generator network
		self.generator = self.build_generator()
		self.compile_generator(self.generator)

		# If training mode build combined GAN network
		if self.training:
			self.vgg = self.build_vgg()
			self.compile_vgg(self.vgg)
			self.discriminator = self.build_discriminator()
			self.compile_discriminator(self.discriminator)
			self.srgan = self.build_srgan()
			self.compile_srgan(self.srgan)

		#print(self.generator.summary())

	def save_weights(self, path="./weights/"):
		"""Save generator and discriminator networks weights"""
		os.makedirs(f'{path}', exist_ok=True)
		self.generator.save_weights(f"{path}/{self.upscaling_factor}X_generator.h5")
		self.discriminator.save_weights(f"{path}/{self.upscaling_factor}X_discriminator.h5")

	def load_weights(self, generator_weights_path=None, discriminator_weights_path=None):
		"""Load generator and/or discriminator networks weights"""
		
		if generator_weights_path:
			self.generator.load_weights(generator_weights_path)
		if discriminator_weights_path:
			self.discriminator.load_weights(discriminator_weights_path)

	def SubpixelConv2D(self, scale, name):
		""" Create pixel shuffle """

		#def subpixel_shape(input_shape):
			#dims = [input_shape[0],
			#None if input_shape[1] is None else input_shape[1] * scale,
			#None if input_shape[2] is None else input_shape[2] * scale,
			#int(input_shape[3] / (scale ** 2))]
			#output_shape = tuple(dims)
			#return output_shape

		def subpixel(x):
			return tf.nn.depth_to_space(x, scale)

		return Lambda(subpixel, name=name)

	def preprocess_vgg(self, x):
		"""Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
		if isinstance(x, np.ndarray):
			return preprocess_input((x+1)*127.5)
		else:            
			return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x) 
	
	def build_vgg(self):
		"""
		Build a pre-trained VGG19 model that outputs images features maps
		which will be used for perceptual loss
		"""

		# Extract features from last conv layer of vgg network
		vgg = VGG19(weights="imagenet", include_top=False, input_shape=self.hr_shape) # TODO: check include_top
		vgg.outputs = vgg.layers[9].output

		# Model build
		model = Model(inputs=vgg.input, outputs=vgg.outputs)
		model.trainable = False

		return model

	def build_generator(self, residual_blocks=16):
		""" Build a generator network """

		# TODO: try ESRGAN (without batch normalization)
		def residual_block(input, filters, number):
			x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=f'res_block{number}_conv1')(input)
			x = BatchNormalization(momentum=0.8, name=f'res_block{number}_batch_normalization1')(x)
			x = PReLU(shared_axes=[1,2], name=f'res_block{number}_PReLU1')(x)
			x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=f'res_block{number}_conv2')(x)
			x = BatchNormalization(momentum=0.8, name=f'res_block{number}_batch_normalization2')(x)
			x = Add(name=f'res_block{number}_add')([x, input])
			
			return x

		# TODO: try changing the order of PReLU and SubpixelConv2D
		def upsample(x, number):
			x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name=f'upSample_Conv2D{number}')(x)
			x = self.SubpixelConv2D(scale=2, name=f'upSample_SubPixel{number}')(x)
			x = PReLU(shared_axes=[1,2], name=f'upSample_PReLU{number}')(x)

			return x
		
		# Input - low-resolution image
		# shape is not predefined so that generator can be used for imgs with various resolution
		lr_img = Input(shape=(None, None, 3)) 

		# Pre-residual 
		c1 = Conv2D(filters=self.g_filters, kernel_size=9, strides=1, padding='same', name='pre_res_Conv2D')(lr_img)
		c1 = PReLU(shared_axes=[1,2], name='pre_res_PReLU')(c1)

		# Residual blocks
		r = residual_block(input=c1, filters=self.g_filters, number=1)
		for i in range(residual_blocks - 1):
			r = residual_block(r, self.g_filters, i+2)

		# Post-residual
		c2 = Conv2D(filters=self.g_filters, kernel_size=3, strides=1, padding='same', name='post_res_Conv2D')(r)
		c2 = BatchNormalization(momentum=0.8, name='post_res_batch_normalization')(c2)
		c2 = Add(name='post_res_add')([c2, c1])

		# Upsampling (source of number of upsampling layers: Smart an Sustainable Intelligent Systems, Namita Gupta, 1.2.1.1, p.6)
		u = upsample(c2, 1)
		if self.upscaling_factor > 2:
			for i in range(ceil(self.upscaling_factor**(1/2)) - 1):
				u = upsample(u, i+2)

		# Generate super resolution image
		sr_img = Conv2D(filters=self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u)

		# Create the model
		model = Model(inputs=lr_img, outputs=sr_img)

		return model

	def build_discriminator(self):
		""" Build a discriminator network """
		
		def conv2d_block(input, filters, strides=1, bn=True):
			x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(input)
			x = LeakyReLU(alpha=0.2)(x)
			if bn:
				x = BatchNormalization(momentum=0.8)(x)
			return x

		# Input - high-resolution image
		img = Input(shape=self.hr_shape)

		# Discriminator blocks
		d = conv2d_block(input=img, filters=self.d_filters, bn=False)
		d = conv2d_block(d, self.d_filters, strides=2)
		d = conv2d_block(d, self.d_filters*2)
		d = conv2d_block(d, self.d_filters*2, strides=2)
		d = conv2d_block(d, self.d_filters*4)
		d = conv2d_block(d, self.d_filters*4, strides=2)
		d = conv2d_block(d, self.d_filters*8)
		d = conv2d_block(d, self.d_filters*8, strides=2)

		d = Dense(self.d_filters*16)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Dense(1, activation='sigmoid')(d)

		# Create the model
		model = Model(inputs=img, outputs=d)

		return model

	def build_srgan(self):
		""" Build a complete SRGAN network """

		# Input - low-resolution image
		lr_img = Input(shape=self.lr_shape)

		# Generate super-resolution image
		sr_img = self.generator(lr_img)
		generated_features = self.vgg(sr_img)#vgg(preprocess_input((sr_img+1)*127.5))

		# In combined model only generator is trainable
		self.discriminator.trainable = False

		# Discriminator determines validity of generated super-resolution image
		sr_validation = self.discriminator(sr_img)

		# Change the names for outputs
		generated_features = Lambda(lambda x: x, name='Content')(generated_features)
		sr_validation = Lambda(lambda x: x, name='Adversarial')(sr_validation)

		# Create the model
		model = Model(inputs=lr_img, outputs=[sr_validation, generated_features])

		return model

	def compile_vgg(self, model):
		""" Compile the vgg network """

		model.compile(
			loss=self.vgg_loss,
			optimizer=Adam(self.vgg_lr, 0.5),
			metrics=['accuracy']
		)

	def compile_generator(self, model):
		""" Compile the generator """

		model.compile(
			loss=self.gen_loss,
			optimizer=Adam(self.gen_lr, 0.5),
			metrics=['mse']
		)

	def compile_discriminator(self, model):
		""" Compile the discriminator """

		model.compile(
			loss=self.dis_loss,
			optimizer=Adam(self.dis_lr, 0.5),
			metrics=['accuracy']
		)

	def compile_srgan(self, model):
		""" Compile combined GAN model """

		model.compile(
            loss=[self.dis_loss, self.gen_loss],
            loss_weights=self.loss_weights,
			optimizer=Adam(self.gen_lr, 0.5)
		)

	def test_images(self, epoch, data_loader, batch_size=1, path="./test_images/"):
		""" Test generator on unseen images """

		os.makedirs(f'{path}/generated/', exist_ok=True)
		hr_imgs, lr_imgs = data_loader.load_data(batch_size=batch_size, testing_data=True)
		sr_imgs = self.generator.predict(lr_imgs)
			
		# Rescale images 0 - 1
		lr_imgs = 0.5 * lr_imgs + 0.5
		hr_imgs = 0.5 * hr_imgs + 0.5
		sr_imgs = 0.5 * sr_imgs + 0.5
			
		# Save generated images and the high resolution originals
		titles = ['Low-resolution', 'High-resolution', 'Super-resolution']

		for batch in range(batch_size):
			fig, axs = plt.subplots(1, 3, figsize=(30,10))
			for col, image in enumerate([lr_imgs, hr_imgs, sr_imgs]):
				axs[col].imshow(image[batch])
				axs[col].set_title(titles[col])
				axs[col].axis('off')
				#os.makedirs(f"{path}/generated/", exist_ok=True)
				if batch_size > 1:
					fig.savefig(f"{path}/generated/{epoch+1}_epoch_{batch}.png")
				else: 
					fig.savefig(f"{path}/generated/{epoch+1}_epoch.png")
			plt.close()
		
	def train(self,
		epochs,
		datapath,
		test_datapath=None,
		test_interval=50,
		batch_size=1,
		weights_path='./weights/',
		print_frequency=1
	):
		""" 
		Train GAN network

		:param int epoch: number of epochs
		:param string datapath: dataset path
		:param int batch_size: size of mini batch
		:param strig weights_path: path where network weights are saved
		:param int print_frequency: how often training progress is being printed
		"""

		if not self.training:
			print("Model is not in training mode!")
			return None

		data_loader = DataLoader(
			datapath=datapath, 
			img_res=(self.hr_height, self.hr_width), 
			scale=self.upscaling_factor
		)

		if test_datapath:
			test_data_loader = DataLoader(
				datapath=test_datapath, 
				img_res=(self.hr_height, self.hr_width), 
				scale=self.upscaling_factor
			)

		loss = {"G": [], "D": []}

		# Dynamic learning rate
		old_lr = K.get_value(self.srgan.optimizer.learning_rate)
		if old_lr < 5e-5:
			new_lr = old_lr * 0.9999
			K.set_value(self.srgan.optimizer.learning_rate, new_lr)
		
		# Training start time
		start_time = datetime.datetime.now()

		for epoch in range(epochs):

			# -----------------------
			#	TRAIN DISCRIMINATOR
			# -----------------------
			discriminator_output_shape = list(self.discriminator.output_shape)
			discriminator_output_shape[0] = batch_size
			discriminator_output_shape = tuple(discriminator_output_shape)
			
			# Sample images
			hr_imgs, lr_imgs = data_loader.load_data(batch_size)
			# Generated super-resolution images
			sr_imgs = self.generator.predict(lr_imgs)

			# Valid / fake targets for discriminator
			real = np.ones(discriminator_output_shape)
			fake = np.zeros(discriminator_output_shape)

			# Train the discriminator on mini batch
			real_loss = self.discriminator.train_on_batch(hr_imgs, real)
			fake_loss = self.discriminator.train_on_batch(sr_imgs, fake)
			discriminator_loss = 0.5 * (np.add(real_loss, fake_loss))

			# -------------------
			#	TRAIN GENERATOR
			# -------------------

			# Sample images
			hr_imgs, lr_imgs = data_loader.load_data(batch_size)

			# Target labels of the generated images
			real = np.ones(discriminator_output_shape)

			# Extracted real image features
			imgs_features = self.vgg.predict(hr_imgs)

			# Train the generator
			generator_loss = self.srgan.train_on_batch(lr_imgs, [real, imgs_features])

			# Generator and discriminator losses and their means
			loss["G"].append(generator_loss[0])
			loss["D"].append(discriminator_loss[0])

			gen_avg_loss = np.array(loss["G"]).mean(axis=0)
			dis_avg_loss = np.array(loss["D"]).mean(axis=0)

			elapsed_time = datetime.datetime.now() - start_time

			if epoch % print_frequency == 0:
				print(f"{epoch+1}/{epochs} epoch time: {elapsed_time} | Generator loss: {gen_avg_loss:.2f} | Discriminator loss: {dis_avg_loss:.4f}")
				
			if epoch==epochs-1 and weights_path:
				self.save_weights(weights_path)

			if (epoch % test_interval == 0) and test_datapath:
				self.test_images(epoch, test_data_loader)
