from __future__ import print_function

import numpy as np

class AddCoordsNp():
	"""Add coords to a tensor"""
	def __init__(self, x_dim=64, y_dim=64, with_r=False):
		self.x_dim = x_dim	# 1080
		self.y_dim = y_dim	# 1920
		self.with_r = with_r

	def call(self):
		"""
			input_tensor: (batch, x_dim, y_dim, c)
		"""
		#batch_size_tensor = np.shape(input_tensor)[0]

		xx_ones = np.ones([self.x_dim], dtype=np.int32)	# shape = (val_h,)
		xx_ones = np.expand_dims(xx_ones, 1)				# shape = (val_h, 1)

		xx_range = np.expand_dims(np.arange(self.y_dim), 0)	# shape = (1, val_w)
		#xx_range = np.expand_dims(xx_range, 1)

		xx_channel = np.matmul(xx_ones, xx_range)		# 每一行是：0, ..., val_w-1	shape = (val_h, val_w)
		xx_channel = np.expand_dims(xx_channel, -1)		# shape = (val_h, val_w, 1)


		yy_ones = np.ones([self.y_dim], dtype=np.int32)	# shape = (val_w,)
		yy_ones = np.expand_dims(yy_ones, 0)				# shape = (1, val_w)

		yy_range = np.expand_dims(np.arange(self.x_dim), 1)	# shape = (val_h, 1)
		#yy_range = np.expand_dims(yy_range, -1)

		yy_channel = np.matmul(yy_range, yy_ones)		# 每一行是：0, ..., val_h-1		shape = (val_h, val_w)
		yy_channel = np.expand_dims(yy_channel, -1)		# shape = (val_h, val_w, 1)


		xx_channel = xx_channel.astype('float32') / (self.y_dim - 1)	# normalization
		yy_channel = yy_channel.astype('float32') / (self.x_dim - 1)

		# 归一化到 -1, 1
		xx_channel = xx_channel*2 - 1	# centralization
		yy_channel = yy_channel*2 - 1
	

		#xx_channel = xx_channel.repeat(batch_size_tensor, axis=0)
		#yy_channel = yy_channel.repeat(batch_size_tensor, axis=0)

		ret = np.concatenate([xx_channel, yy_channel], axis=-1)	# shape = (val_h, val_w, 2)

		if self.with_r:
			rr = np.sqrt( np.square(xx_channel-0.5) + np.square(yy_channel-0.5))
			ret = np.concatenate([ret, rr], axis=-1)

		return ret
