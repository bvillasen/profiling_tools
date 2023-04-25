import numpy as np
import pandas as pd


counters = { 
  'FP16_FLOPs':     [ 'SQ_INSTS_VALU_ADD_F16', 'SQ_INSTS_VALU_MUL_F16', 'SQ_INSTS_VALU_FMA_F16', 'SQ_INSTS_VALU_TRANS_F16'],
  'FP32_FLOPs':     [ 'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32', 'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32'],
  'FP64_FLOPs':     [ 'SQ_INSTS_VALU_ADD_F64', 'SQ_INSTS_VALU_MUL_F64', 'SQ_INSTS_VALU_FMA_F64', 'SQ_INSTS_VALU_TRANS_F64'],
  'MOPs':           [ 'SQ_INSTS_VALU_MFMA_MOPS_F16', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'SQ_INSTS_VALU_MFMA_MOPS_F32', 'SQ_INSTS_VALU_MFMA_MOPS_F64'],  
  'HBM_Bandwidth':  [ 'TCC_EA_RDREQ_sum', 'TCC_EA_RDREQ_32B_sum', 'TCC_EA_WRREQ_sum', 'TCC_EA_WRREQ_64B_sum' ],
  'LDS_Bandwidth':  [ 'SQ_LDS_IDX_ACTIVE', 'SQ_LDS_BANK_CONFLICT' ],
  'L2 Bandwidth':   [ 'TCP_TCC_READ_REQ_sum', 'TCP_TCC_WRITE_REQ_sum', 'TCP_TCC_ATOMIC_WITH_RET_REQ_sum', 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'],
  'VLID_Bandwidth': [ 'TCP_TOTAL_CACHE_ACCESSES_sum'],
  'Utilization_0':  [ 'Wavefronts', 'VALUInsts', 'VFetchInsts', 'VALUUtilization', 'VALUBusy', 'WriteSize'],
  'Utilization_1':  [ 'SALUInsts', 'SFetchInsts', 'LDSInsts', 'FlatLDSInsts', 'GDSInsts', 'SALUBusy', 'FetchSize'],
  'Utilization_2':  [ 'L2CacheHit', 'MemUnitBusy', 'MemUnitStalled', 'WriteUnitStalled', 'ALUStalledByLDS', 'LDSBankConflict']
}

def load_rocprof_results( file_name, load_with='pandas' ):
  print( f'Loading file: {file_name}' )
  if load_with == 'pandas':
    df = pd.read_csv( file_name )
    return df

def load_rocprof_stats( input_dir, file_name='results.stats.csv', mpi=False, mpi_rank=0, time_units='msecs' ):
  if mpi: file_name = f'{input_dir}/rank_{mpi_rank}/{file_name}'
  else: file_name = f'{input_dir}/{file_name}'
  # print( f'Loading file: {file_name}')
  data_set = load_rocprof_results( file_name )
  profile_data = {}
  data = data_set.values
  for indx,line in enumerate(data):
    name, ncalls, time_total, time_avrg, percent = line
    if time_units == 'msecs': 
      time_total*= 1e-6
      time_avrg*= 1e-6 
    profile_data[indx] = { 'name':name, 'n_calls':ncalls, 'time_total':time_total, 'time_avrg':time_avrg, 'percent':percent, 'time_units':time_units }
  return profile_data

def get_kernel_names( data_set, ignore_kd=False ):
  kernel_names = list(set(list(data_set['KernelName'])))
  if ignore_kd:
    kernel_names = [ name for name in kernel_names if name.find('.kd') == -1 ]
  return kernel_names


def get_kernel_statistics( data_set, kernel_name, rocprof_stats=None, print_warnings=False, first_call_tolerance=1.5 ):
  print (f' Getting kernel statistics: {kernel_name} ')
  kernel_data = data_set.loc[data_set['KernelName']==kernel_name]
  n_calls, n_data = kernel_data.shape
  kernel_stats = {'name':kernel_name, 'n_calls':n_calls }
 
  # Use the statictics collected separatelly by rocprof 
  # instead of the duration in the counters file since 
  # counter collection adds overhead and decreases performance
  if rocprof_stats is not None:
    rocprof_kernel = rocprof_stats.loc[rocprof_stats['Name']==kernel_name]
    indx_0 = rocprof_kernel.index[0]
    Calls = rocprof_kernel['Calls'][indx_0]
    TotalDurationNs = rocprof_kernel['TotalDurationNs'][indx_0]
    AverageNs = rocprof_kernel['AverageNs'][indx_0]
    Percentage = rocprof_kernel['Percentage'][indx_0]
    kernel_stats['time_total_Ns'] = TotalDurationNs 
    kernel_stats['time_mean_Ns']  = AverageNs
    kernel_stats['Percentage']  = Percentage
    if Calls != n_calls:
      print( f"  ERROR: mismath in number of kernel calls: {n_calls}   {Calls}")

  # If no statistics are provided, then use the duration 
  # in the counters file. Note that this is not recomended 
  # since counter collection adds overhead and decreases performance 
  else:
    if 'durationNs' in kernel_data:
      duration = kernel_data['DurationNs'] 
    else:
      duration =  kernel_data['EndNs'] - kernel_data['BeginNs']
    indx_0 = kernel_data.index[0]
    duration_corrected = duration.drop(index=indx_0)
    duration_mean = duration_corrected.mean()
    duration_0_fraction = duration[indx_0] / duration_mean
    if duration_0_fraction > first_call_tolerance:
      print( f'  WARNING: Skiping first call which takes {duration_0_fraction} times longer than average.' )
      duration = duration_corrected
      kernel_data  = kernel_data.drop(index=indx_0)
    kernel_stats['duration_Ns'] = duration    
    kernel_stats['time_total_Ns'] = duration.sum() 
    kernel_stats['time_mean_Ns']  = duration.mean()

  kernel_stats['counters_mean'] = {}
  for counter_type in counters:
    for counter_name in counters[counter_type]:
      counter_data = kernel_data[counter_name].values
      counter_mean = counter_data.mean()
      if counter_mean != counter_data[0] and print_warnings: 
        print(f'WARNING: Counter {counter_name} has multiple values:  mean: {counter_mean}   data: {counter_data[0]} ')
      kernel_stats['counters_mean'][counter_name] = counter_mean 
  return kernel_stats

def get_all_kernel_statictics( data_set, rocprof_stats=None, print_warnings=False, first_call_tolerance=1.5  ):
  print( '\nComputing all kernel statistics' )
  kernel_names = get_kernel_names( data_set )
  statistics = {}
  time_total = 0
  for kernel_id, kernel_name in enumerate(kernel_names):
    # print(rocprof_stats)
    kernel_stats = get_kernel_statistics(data_set, kernel_name, rocprof_stats=rocprof_stats, print_warnings=print_warnings, first_call_tolerance=first_call_tolerance ) 
    statistics[kernel_id] = kernel_stats  
    time_total += kernel_stats['time_total_Ns']
  time_fractions = []  
  for kernel_id, kernel_name in enumerate(kernel_names):
    statistics[kernel_id]['time_fraction'] = statistics[kernel_id]['time_total_Ns'] / time_total
    time_fractions.append( statistics[kernel_id]['time_fraction'] )
  time_fractions = np.array(time_fractions)
  sorted_ids = np.argsort(time_fractions)
  sorted_ids = sorted_ids[::-1]
  statistics_sorted = {}
  for id, sorted_id in enumerate(sorted_ids): 
    statistics_sorted[id] = statistics[sorted_id]  
  return statistics_sorted

skip_avrg_keys = ['counters_mean']
def average_mpi_values( mpi_values ):
  kernel_ids = mpi_values[0].keys() 
  keys = mpi_values[0][0].keys()
  values_lists = {}
  n_mpi = len( mpi_values )
  for kernel_id in kernel_ids:
    values_lists[kernel_id] = {}
    for key in keys:
      if key == 'name': continue
      if key == 'time_units': continue
      if key == 'counters_mean': continue
      if key not in values_lists[kernel_id]: values_lists[kernel_id][key] = []
      for mpi_id in range(n_mpi):
        mpi_value = mpi_values[mpi_id][kernel_id][key]
        values_lists[kernel_id][key].append( mpi_value )
  avrg_values = {}
  for kernel_id in kernel_ids:
    avrg_values[kernel_id] = {}
    for key in keys:
      if key in skip_avrg_keys: continue
      if key in ['name', 'time_units']: 
        val = mpi_values[0][kernel_id][key] 
        avrg_values[kernel_id][key] = val
        continue
      key_values = np.array(values_lists[kernel_id][key])  
      avrg_values[kernel_id][key] = { 'mean': key_values.mean(), 'sigma':key_values.std() }
  return avrg_values

