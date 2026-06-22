# PolyDis is a deterministic open-source code for atomic
# displacement calculations in arbitrary polyatomic materials
# based on the Lindhard theoretical framework.
#
# PolyDis is writen in and runs with Python3
#
# Copyright (c) 2026 Shengli Chen
# Sun Yat-sen University
# MIT License
#
#
# The calculations results are automatically stored in an ASCII
# file named {material_name}_disp.dat, where {material_name} is
# the first parameter in PolyDis.solve_polyatomic_displacement().
# {material_name} can be anything because it is used only for 
# saving the computational results.
#
#
# This package includes the following files.

- list_elements.txt 
	The file includes basic atomic data.
	This file is however not mandatory, users are allowed
	to assign any data, whether realistic or vitural.

- PolyDisp.py
	The main Python3 source file of the program.
	It includes all functions and numerical methods.
	Instructions are given in the source file.

- templateInput.py
	A template input file for calculations of FeCrAl triatomic system.
	Examples of extracting the computational results are also given.

- tmp_disp.dat
	Output file for the input template.
	"tmp" corresponds to the first parameter in the input.
	
