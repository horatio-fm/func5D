import sys

import os
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pylab import *
from matplotlib.widgets import Slider

os.chdir("c:/Users/horatio/PycharmProjects/func5D/")

def save_object (obj, filename):
	import cPickle as pickle
	# import pickle
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, -1)

def load_object(filename):
	import cPickle as pickle
	# import pickle
	with open(filename, 'rb') as output:
		obj = pickle.load(output)
	return obj	

# filename_multidim_var = "fitness_func_values_python_list1D_5D_dim_size_21_angle_range_4.obj"
filename_multidim_var = "fmodel_structure__fitness_func_values_5D_dim_size_21_angle_range_4.obj"
fitness_func_values_5D = load_object(filename_multidim_var)

def extract_value(var, filename_multidim_var):
	number_start = filename_multidim_var.find(var) + 1 + len(var)
	if number_start == -1:
		raise "substring not found"
	number_end = number_start + filename_multidim_var[filename_multidim_var.find(var) + 1 + len(var):].find("_")
	if number_end == number_start - 1:
		number_end = number_start + filename_multidim_var[filename_multidim_var.find(var) + 1 + len(var):].find(".")
		if number_end == number_start - 1:
			raise "substring not found"
	return int(filename_multidim_var[number_start:number_end])
	
dim_size = extract_value("dim_size", filename_multidim_var)	
angle_range = extract_value("angle_range", filename_multidim_var)	

phi_theta_psy_x_y = np.zeros((dim_size, dim_size, dim_size, dim_size, dim_size))

projection_location = (15, 30,20,1,1)

phi_ax_vals = np.linspace(-angle_range,angle_range,dim_size) + projection_location[0]
theta_ax_vals = np.linspace(-angle_range,angle_range,dim_size) + projection_location[1]
psy_ax_vals = np.linspace(-angle_range,angle_range,dim_size) + projection_location[2]
shift_x_ax_vals = np.linspace(-angle_range,angle_range,dim_size) + projection_location[3]
shift_y_ax_vals = np.linspace(-angle_range,angle_range,dim_size) + projection_location[4]

labels_axes = ["phi", "theta", "psy", "shift_x", "shift_y"]
vals_axes = [phi_ax_vals, theta_ax_vals, psy_ax_vals, shift_x_ax_vals, shift_y_ax_vals]

def combinations_of_n_taken_by_k(n, k):
	from fractions import Fraction
	return int(reduce(lambda x, y: x * y, (Fraction(n-i, i+1) for i in range(k)), 1))

combinations_of_n_taken_by_k(5,2)

fitness_func_values_5D = fitness_func_values_5D.reshape(dim_size,dim_size, dim_size,dim_size, dim_size)

def plot_arange(figure_index):
	import matplotlib.pyplot as plt
	
	def pos_list_f():
		# yield (0, 0, 500, 460)
		x_size = 500
		y_size = 460
		x_buffer_size = 0
		y_buffer_size = 50
		for j in range(3):
			for i in range(5):
				if j == 2:
					yield  i*(x_size + x_buffer_size), j*(y_size + y_buffer_size) - 40, x_size, y_size
					continue
				yield  i*(x_size + x_buffer_size), j*(y_size + y_buffer_size), x_size, y_size 
	
	if figure_index == -1:
		return list(pos_list_f())
	
	if "first_time" not in plot_arange.__dict__:
		plot_arange.first_time = 1
		plot_arange(figure_index)
	
	pos_list1 = list(pos_list_f())
	plt.show(block=False)
	mngr = plt.get_current_fig_manager()
	pos = pos_list1[figure_index]
	mngr.window.setGeometry(*pos)
	mngr.window.raise_()

from numpy import unravel_index

plot_count = 0
plt.close("all")

import itertools
all_combinations = list(itertools.combinations(range(5),2))

list_var_sliders = []
list_var_axes = []

list_var_sliders_vals_indices = [[2,3,4], [3,4,1], [2,4,1,], [2,3,1], [0,3,4], [0,2,4], [0,2,3], [0,4,1], [0,3,1], [0,2,1]]

def update(val):
	
	# list_var_sliders_vals = []
	for i in range(len(list_var_axes)):
		for j in range(len(list_var_sliders[i])):
			if list_var_sliders_vals[i][j] != list_var_sliders[i][j].val:
				list_var_sliders_vals[i][j] = val

				ax = ax_list[i]

				oldcol = ax_list[i].collections
				for old1 in oldcol:
					ax.collections.remove(old1)
				
				mycomb = all_combinations[i]
				X, Y = np.meshgrid(vals_axes[mycomb[0]],vals_axes[mycomb[1]])
				X = X.transpose()
				Y = Y.transpose()
				index = []
				for i1 in range(5):
					if i1 in mycomb:
						index.append(np.s_[:])
					else:
						iii = list_var_sliders_vals_indices[i].index(i1)
						index.append(int(list_var_sliders_vals[i][iii]))
								
				index = tuple(index)
				Z = fitness_func_values_5D[index]
				max_idx = unravel_index(Z.argmax(), Z.shape)
			
				# wframe[i] = ax.plot_wireframe(X, Y, Z)
				ax.plot_wireframe(X, Y, Z)
				
				ax.scatter(X[int(X.shape[0]/2)][int(X.shape[1]/2)], Y[int(Y.shape[0]/2)][int(Y.shape[1]/2)], Z[int(Z.shape[0]/2)][int(Y.shape[1]/2)], color = "red")
				ax.scatter(X[max_idx[0]][max_idx[1]], Y[max_idx[0]][max_idx[1]], Z[max_idx[0]][max_idx[1]], color = "green")

				plt.pause(.001)				
							
				return
	

fig_list = []
ax_list = []
wframe = [None]*10
for comb_idx, mycomb in enumerate(all_combinations):
	X, Y = np.meshgrid(vals_axes[mycomb[0]],vals_axes[mycomb[1]])
	X = X.transpose()
	Y = Y.transpose()
	index = []
	for i in range(5):
		if i in mycomb:
			index.append(np.s_[:])
		else:
			index.append(0)
				
	index = tuple(index)
	Z = fitness_func_values_5D[index]
	max_idx = unravel_index(Z.argmax(), Z.shape)

	fig = plt.figure()
	fig_list.append(fig)
	ax = fig.add_subplot(111, projection='3d')
	ax_list.append(ax)
	plt.hold(True)

	ax.plot_wireframe(X, Y, Z)
	
	ax.scatter(X[int(X.shape[0]/2)][int(X.shape[1]/2)], Y[int(Y.shape[0]/2)][int(Y.shape[1]/2)], Z[int(Z.shape[0]/2)][int(Y.shape[1]/2)], color = "red")
	ax.scatter(X[max_idx[0]][max_idx[1]], Y[max_idx[0]][max_idx[1]], Z[max_idx[0]][max_idx[1]], color = "green")
	
	ax.set_xlabel(labels_axes[mycomb[0]])
	ax.set_ylabel(labels_axes[mycomb[1]])

	# ax.view_init(elev=65, azim=-34)
	ax.view_init(elev=30, azim=-34)
	
	on_the_side_labels = list(set(labels_axes) - set([labels_axes[mycomb[0]], labels_axes[mycomb[1]]]))
	# on_the_side_labels = list(set(labels_axes) - set([labels_axes[0], labels_axes[1]]))
	on_the_side_labels.sort()
	
	axcolor = 'lightgoldenrodyellow'
	
	var_axes = []
	for i in range(3):
		var_axes.append(axes([0.1, 0.03*i + 0.02, 0.8, 0.01], axisbg=axcolor))

	var_sliders = []
	for i in range(3):
		var_sliders.append(Slider(var_axes[i], on_the_side_labels[i], 0.1, 20.0, valinit=.5))
		var_sliders[-1].on_changed(update)
	
	list_var_sliders.append(var_sliders)
	list_var_axes.append(var_axes)
	
	# show(block=False)
	plot_arange(plot_count)
	if plot_count == 0:
		plot_arange(plot_count)
	
	plot_count += 1
	
	# draw(block=False)


list_var_sliders_vals = []
for i in range(len(list_var_axes)):
	list_var_sliders_vals.append([])
	for j in range(len(list_var_sliders[i])):
		list_var_sliders_vals[i].append(list_var_sliders[i][j].val)
	
	
show()


