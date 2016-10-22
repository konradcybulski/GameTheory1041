# GameTheory1041
Repository for work regarding the 1041 undergrad research project.

Working with the simulation:

	In order to run the simulation with predefined parameters,
	using Python simply run the SimulationMain.py or one of the
	main.py or main_communication.py files. These are the files
	which contain the main functions for calling a series of 
	simulations (N simulations for N processors on a machine).
	
	For SimulationMain, the parameters for the simulation are 
	defined in the file SantosSantosPacheco.py in functions
	called by the main files. Changing these parameters will 
	directly alter the simulation and the results will reflect
	the changed paramters.
	
	The return value is the average cooperation index of the 
	simulation ran. The number of generations recommended is 
	3 * 10^5, but a larger number is required for a larger 
	population size. In order to find the most accurate average
	cooperation index, the number of recommended runs is 100,
	or 100 divided by the number of processors and the average
	of the result from each processor.
	
	For efficiency and speed, the recommended program is the 
	Cython program which can be found in 
	Python_Resource/Cython_resources. Here the source code for
	the simulation can be found in simulation_instance.pyx and 
	for the variable assessment error (communication delay) 
	program simulation_instance_comms.pyx. Each of these must 
	be built using the setup.py file which will output to a 
	subfolder Cython_resources within Cython_resources. In
	order to utilize these changes, move these files from 
	that subfolder into the subfolder 'build' and rename them
	using the current naming meta, otherwise change the 
	import statements in the main files.