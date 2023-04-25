import sys, os, time
import numpy as np
root_dir = os.getcwd()
sub_directories = [x[0] for x in os.walk(root_dir) if x[0].find('.git')<0 ]
sys.path.extend(sub_directories)
import profile_functions as pf
import roofline_functions as roof
import importlib
importlib.reload(roof) 
importlib.reload(pf) 

work_dir = '/mnt/c/Users/bvillase/work/'


# Application parameters
problem = 1
n_qubit, n_mpi = 31, 2
profile_name = f'nqubit{n_qubit}_nmpi{n_mpi}'

input_dir    = work_dir + f'tests/quest/p{problem}/mi250x/profile_{profile_name}/'
stats_dir    = input_dir + f'rocprof_stats/'
counters_dir = input_dir + f'rocprof_counters/'


mpi_statistics, mpi_roofline = [], [] 
for mpi_rank in range(n_mpi):

  # Load the timing stats produced by rocprof
  stats_file_name = stats_dir + f'rank_{mpi_rank}/results.stats.csv'
  rocprof_stats = pf.load_rocprof_results( stats_file_name, load_with='pandas' ) 

  # Load the counters dataset
  counters_file_name = counters_dir + f'rank_{mpi_rank}/results.csv'
  data_set = pf.load_rocprof_results( counters_file_name, load_with='pandas' )
  kernel_names = pf.get_kernel_names( data_set ) 

  # Get counter statistics and roofline metrics
  kernel_statistics = pf.get_all_kernel_statictics( data_set, rocprof_stats=rocprof_stats, print_warnings=False )
  roofline_metrics = roof.get_roofline_metrics( kernel_statistics )
  mpi_statistics.append( kernel_statistics )
  mpi_roofline.append( roofline_metrics )

#  Get the averaged statistics and metrics over all the MPI ranks
avrg_statistics = pf.average_mpi_values( mpi_statistics )
avrg_roofline = pf.average_mpi_values( mpi_roofline )

#Load roofline bounds
file_name = root_dir + '/tools/roofline_mi250.csv'
roofline_bounds = roof.load_gpu_roofline_bounds( file_name, extrapolate_to_mi250x=True )

# Make the roofline figure
output_dir = '/mnt/c/Users/bvillase/work/projects/quest/'
figure_name = f'roofline_mi250x_p1_{profile_name}_nqbits_{n_qubit}_nmpi_{n_mpi}'
roof.plot_roofline( avrg_statistics, avrg_roofline, roofline_bounds, 
                    output_dir, averaged_statistics=True, figure_name=figure_name, 
                    total_to_cover=1.0, types=['HBM'], AI_bounds=[1e-3, 1e2]   )



