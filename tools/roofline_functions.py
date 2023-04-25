import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

def get_kernel_roofline_metrics( statistics, kernel_id, append_to_statistics=False, FLOPS_min=1e-2 ):
  name = statistics[kernel_id]['name']
  print (f' Getting kernel roofline metrics: {name}')
  n_calls = statistics[kernel_id]['n_calls']
  time = statistics[kernel_id]['time_mean_Ns']
  time_total = statistics[kernel_id]['time_total_Ns']
  percentage = statistics[kernel_id]['Percentage']
  kc = statistics[kernel_id]['counters_mean']
  F_16 = kc['SQ_INSTS_VALU_ADD_F16'] + kc['SQ_INSTS_VALU_MUL_F16'] + kc['SQ_INSTS_VALU_TRANS_F16'] + 2*kc['SQ_INSTS_VALU_FMA_F16']
  F_32 = kc['SQ_INSTS_VALU_ADD_F32'] + kc['SQ_INSTS_VALU_MUL_F32'] + kc['SQ_INSTS_VALU_TRANS_F32'] + 2*kc['SQ_INSTS_VALU_FMA_F32']
  F_64 = kc['SQ_INSTS_VALU_ADD_F64'] + kc['SQ_INSTS_VALU_MUL_F64'] + kc['SQ_INSTS_VALU_TRANS_F64'] + 2*kc['SQ_INSTS_VALU_FMA_F64']   
  MOPS = kc['SQ_INSTS_VALU_MFMA_MOPS_F16'] + kc['SQ_INSTS_VALU_MFMA_MOPS_BF16'] + kc['SQ_INSTS_VALU_MFMA_MOPS_F32'] + kc['SQ_INSTS_VALU_MFMA_MOPS_F64']
  FLOPS   = 64 * ( F_16 + F_32 + F_64 ) + 512 * MOPS
  LDS_BW  = 32 * 4 * ( kc['SQ_LDS_IDX_ACTIVE'] - kc['SQ_LDS_BANK_CONFLICT'] )
  vL1D_BW = 64 * kc['TCP_TOTAL_CACHE_ACCESSES_sum']
  L2_BW   = 64 * ( kc['TCP_TCC_READ_REQ_sum'] + kc['TCP_TCC_WRITE_REQ_sum'] + kc['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'] + kc['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum']) 
  HBM_BW_0  = 32 * kc['TCC_EA_RDREQ_32B_sum'] + 64 * ( kc['TCC_EA_RDREQ_sum'] - kc['TCC_EA_RDREQ_32B_sum'] ) \
            + 32 * ( kc['TCC_EA_WRREQ_sum'] - kc['TCC_EA_WRREQ_64B_sum'] ) + 64 * kc['TCC_EA_WRREQ_64B_sum'] 
  # HBM_BW = HBM_BW_0
  HBM_BW_1  = ( kc['WriteSize'] + kc['FetchSize'] ) * 1024  #Counters are in kbytes, convert to bytes
  HBM_BW = HBM_BW_1
 
  FLOPS   /= time * 1e3 # in TFLOPS  (teraflops) the 1e3 factor converts from 1/Ns to Tera
  if FLOPS < FLOPS_min : FLOPS = FLOPS_min 
  time *= 1e-9 #convert from Ns to secs
  LDS_BW  /= time * 1024**4 # in TB/s  
  vL1D_BW /= time * 1024**4 # in TB/s  
  L2_BW   /= time * 1024**4 # in TB/s  
  HBM_BW  /= time * 1024**4 # in TB/s
  AI_HBM = FLOPS / HBM_BW
  if FLOPS == FLOPS_min: 
    rand_frac = 0.1
    AI_HBM *= ( 1 + (2*rand_frac*np.random.rand() - rand_frac) ) 
  AI_LDS = FLOPS / LDS_BW if LDS_BW != 0 else 0
  AI_vL1D = FLOPS / vL1D_BW
  AI_L2 = FLOPS / L2_BW
  roofline_metrics = { 'name':statistics[kernel_id]['name'],
                       'n_calls':n_calls,
                       'time_total': time_total * 1e-9,
                       'percentage': percentage,
                       'FLOPS':FLOPS, 'HBM_BW':HBM_BW, 'LDS_BW':LDS_BW, 
                       'L1_BW':vL1D_BW, 'L2_BW':L2_BW, 'AI_HBM':AI_HBM,
                       'AI_LDS':AI_LDS, 'AI_L2':AI_L2, 'AI_L1':AI_vL1D }
  if append_to_statistics: statistics[kernel_id]['roofline_metrics'] = roofline_metrics
  return roofline_metrics 

def get_roofline_metrics( statistics, append_to_statistics=False ):
  print( '\nComputing all kernel roofline metrics' )
  roofline_metrics = {}
  for kernel_id in statistics: 
    roofline_metrics[kernel_id] = get_kernel_roofline_metrics( statistics, kernel_id, append_to_statistics=append_to_statistics )
  return roofline_metrics


def load_gpu_roofline_bounds( file_name, extrapolate_to_mi250x=False ):
  print ( f'Loading roofline file: {file_name}')
  import pandas as pd
  df = pd.read_csv( file_name )
  df_mean = df.mean()
  FP64Flops = df_mean['FP64Flops']
  if extrapolate_to_mi250x: FP64Flops *= 110/104
  HBM_BW = df_mean['HBMBw']
  LDS_BW = df_mean['LDSBw']
  L2_BW = df_mean['L2Bw']
  L1_BW = df_mean['L1Bw']
  roofline_bounds = { 'FP64Flops':FP64Flops, 'HBM_BW':HBM_BW, 'LDS_BW':LDS_BW, 'L2_BW':L2_BW, 'L1_BW':L1_BW }
  return roofline_bounds




def plot_roofline( kernel_statistics, roofline_metrics, roofline_bounds, 
                  output_dir, averaged_statistics=False, figure_name='roofline', types=['HBM'],
                  total_to_cover = 0.99, skip_kernels=[], AI_bounds=[5e-3, 1e2]   ):

  figure_width = 8
  h_scale_factor = 0.8
  font_size = 12
  legend_size = 8
  fig_text_size = 8
  text_color = 'black'

  nrows, ncols = 1, 1  
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figure_width*ncols,figure_width*nrows*h_scale_factor))
  plt.subplots_adjust( hspace = 0.1, wspace=0.15)

  # Plot roofline bounds
  AI_min, AI_max = AI_bounds
  for type in types:
    AI_range = np.logspace( np.log10(AI_min), np.log10(AI_max), 1000 )
    FP64Flops = roofline_bounds['FP64Flops']
    bw_key = f'{type}_BW' 
    bound_HBM_BW = roofline_bounds[bw_key]
    bound_HBM_Flops = AI_range * bound_HBM_BW
    bound_HBM_Flops[bound_HBM_Flops>FP64Flops] = FP64Flops
    label = f'{type} roofline' 
    ax.plot( AI_range, bound_HBM_Flops, label=label )
    # ax.plot( AI_range, bound_HBM_Flops*0.6 )
    text = f'Achievable FLOP/s = {roofline_bounds["FP64Flops"]/1e3:.1f} TFLOP/s'
    text_x, text_y = 5.5, 2.4e4
    ax.text(text_x, text_y, text, fontsize=fig_text_size)
    # text = f'Peak HBM BW = {roofline_bounds["HBM_BW"]/1e3:.1f} TB/s'
    text = f'Achievable HBM BW = {roofline_bounds["HBM_BW"]/1e3:.1f} TB/s'
    text_x, text_y = 0.1, 1.8e2
    ax.text(text_x, text_y, text, fontsize=fig_text_size, rotation=45)

  for type_indx, type in enumerate(types):
    total_covered = 0
    for kernel_id in roofline_metrics:
      if total_covered > total_to_cover: continue
      metrics = roofline_metrics[kernel_id]
      stats = kernel_statistics[kernel_id]
      name = metrics['name']
      if name != stats['name']: print('ERROR: kernel stats and metrics mismatch')
      
      if averaged_statistics: fraction = stats['time_fraction']['mean']
      else: fraction = stats['time_fraction']

      indx_end = name.find('_kernel')
      name = name[:indx_end]
      if name in skip_kernels: continue
      total_covered += fraction
      percentage = fraction * 100
      label = f'{percentage:.1f}%  {name}'
      if averaged_statistics:
        AI_mean = metrics[f'AI_{type}']['mean']
        flops_mean = metrics['FLOPS']['mean']*1e3 #Convert from TFLOPS/s to GFLOPS/s
        flops_sigma = metrics['FLOPS']['sigma']*1e3
      else:
        AI_mean = metrics[f'AI_{type}']
        flops_mean = metrics['FLOPS']*1e3 #Convert from TFLOPS/s to GFLOPS/s
        flops_sigma = metrics['FLOPS']*1e3
      if type_indx != 0: label = ''
      if averaged_statistics:
        ax.errorbar( AI_mean, flops_mean, yerr=flops_sigma, label=label, fmt='o', color=f'C{kernel_id}', zorder=1)
      else:
        ax.scatter( AI_mean, flops_mean, label=label,  color=f'C{kernel_id}', zorder=1)
         

  ax.legend(loc=2, frameon=False, fontsize=legend_size)
  ax.set_xscale('log')
  ax.set_yscale('log')


  ax.set_ylabel( 'GFLOP/s', fontsize=font_size, color=text_color  )
  ax.set_xlabel( 'Arithmetic Intensity [FLOP/Byte]', fontsize=font_size, color=text_color  )
  figure_name = f'{output_dir}{figure_name}.png'
  fig.savefig( figure_name, bbox_inches='tight', dpi=400, facecolor=fig.get_facecolor() )
  print( f'Saved Figure: {figure_name}' )

