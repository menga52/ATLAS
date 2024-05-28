import numpy as np
from ATLAS.ATLAS.weighted_drift_diffusion2 import weighted_drift_diffusion2
from ATLAS.ATLAS.ATLAS_simulator2 import ATLAS_simulator2

"""
not close to finished
"""
  
def Peanut_MFPT(chart_sim_parameter, RHS_parameter, connectivity, datapath=""):
  Mean_FPT                           = np.zeros((12,10))
  t_final                            = np.zeros((12,10))
  relative_error_FPT                 = np.zeros((8,10))

  for k in range(10):
    print("round",k)
    if datapath != "":
      chart_fileName = datapath+'chart'+str(k)+'.mat'
      TranM_fileName = datapath+'TranM'+str(k)+'.mat'
      FPT_fileName = datapath+'Peanut_FPT'+str(k)+'.mat'
      #load(chart_fileName)
      #load(TranM_fileName)

    mode = 2
    t0 = RHS_parameter["t0"]
    chi_p = RHS_parameter["chi_p"]
    D = RHS_parameter["D"]
    d = RHS_parameter["d"]
    threshold = RHS_parameter["threshold"]
    def weighted_dd2(X0, chart,neigh,nearest, connectivity_indices):
      weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode )
    
    chart_sim_parameter.connectivity      = connectivity 
    chart_sim_parameter.X_int             = chart[chart_sim_parameter.nearest].X_int
    chart_sim_parameter.explore_threshold = 8
    chart_sim_parameter.gap = 1
    # chart_sim_parameter.Nstep = 2*10**7
    X, nearest_store, chart = ATLAS_simulator2(weighted_dd2, chart_sim_parameter,RHS_parameter, simulator_par, chart);           disp(['One single trajectory is simulated of time ', num2str(chart_sim_parameter.Nstep*chart_sim_parameter.dt_s)])

      %% Calculate MFPT

      print('Starting MFPT part')
      N_IC = 15000
      set_well

      disp('First Well: 0.05 level exit 0.02 level')
      [FPT_ori1, t_final(1,k)]              = MFPT_peanut_ori_general(well_threshold1,  X_int_store1, RHS_parameter,chart_angle);

      disp('Fast mode projection')
      option                  = 1;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_sim1, t_final(2,k)]              = MFPT_peanut_sim_general(weighted_dd2, well_threshold1, X_int_store1,nearest_store1, chart,chart_sim_parameter,chart_angle);

      disp('Orthogonal projection')
      option                  = 2; 
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_ort1, t_final(3,k)]              = MFPT_peanut_sim_general(weighted_dd2, well_threshold1, X_int_store1,nearest_store1, chart,chart_sim_parameter,chart_angle);

      %%
      disp('Second Well: -0.05 level exit -0.02 level') 
      [FPT_ori2, t_final(4,k)]              = MFPT_peanut_ori_general(well_threshold2,  X_int_store2,RHS_parameter,chart_angle);

      disp('Fast mode projection')
      option                  = 1;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_sim2, t_final(5,k)]              = MFPT_peanut_sim_general(weighted_dd2, well_threshold2, X_int_store2,nearest_store2, chart,chart_sim_parameter,chart_angle);


      disp('Orthogonal projection')
      option                  = 2; 
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_ort2, t_final(6,k)]              = MFPT_peanut_sim_general(weighted_dd2, well_threshold2, X_int_store2,nearest_store2, chart,chart_sim_parameter,chart_angle);


      %%
      disp('Starting First Well: 0.05 level exit zero line')
      [FPT_ori3, t_final(7,k)]              = MFPT_peanut_ori_zeroline(X_int_store1,RHS_parameter,phi_zero, theta_zero);

      disp('Fast mode projection')
      option                  = 1;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_sim3, t_final(8,k)]              = MFPT_peanut_sim_zeroline(weighted_dd2, X_int_store1, nearest_store1, chart,chart_sim_parameter,phi_zero, theta_zero);

      option                  = 2;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_ort3, t_final(9,k)]              = MFPT_peanut_sim_zeroline(weighted_dd2, X_int_store1, nearest_store1, chart,chart_sim_parameter,phi_zero, theta_zero);

      %%
      disp('Starting Second Well: -0.05 level exit zero line')
      [FPT_ori4, t_final(10,k)]              = MFPT_peanut_ori_zeroline(X_int_store2,RHS_parameter,phi_zero, theta_zero);

      disp('Fast mode projection')
      option                  = 1;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_sim4, t_final(11,k)]              = MFPT_peanut_sim_zeroline(weighted_dd2, X_int_store2, nearest_store2, chart,chart_sim_parameter,phi_zero, theta_zero);

      option                  = 2;
      weighted_dd2            = @(X0, chart,neigh,nearest, connectivity_indices) weighted_drift_diffusion2( X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D,d, threshold, option,mode );
      [FPT_ort4, t_final(12,k)]              = MFPT_peanut_sim_zeroline(weighted_dd2, X_int_store2, nearest_store2, chart,chart_sim_parameter,phi_zero, theta_zero);

      save(FPT_fileName,'FPT_ori1','FPT_ori2','FPT_ori3','FPT_ori4','FPT_sim1','FPT_sim2','FPT_sim3',...
                        'FPT_sim4','FPT_ort1','FPT_ort2','FPT_ort3','FPT_ort4')



    Mean_FPT(1,k)           = mean(FPT_ori1);
    Mean_FPT(2,k)           = mean(FPT_ori2);
    Mean_FPT(3,k)           = mean(FPT_ori3);
    Mean_FPT(4,k)           = mean(FPT_ori4);
    Mean_FPT(5,k)           = mean(FPT_sim1);
    Mean_FPT(6,k)           = mean(FPT_sim2);
    Mean_FPT(7,k)           = mean(FPT_sim3);
    Mean_FPT(8,k)           = mean(FPT_sim4);
    Mean_FPT(9,k)           = mean(FPT_ort1);
    Mean_FPT(10,k)          = mean(FPT_ort2);
    Mean_FPT(11,k)          = mean(FPT_ort3);
    Mean_FPT(12,k)          = mean(FPT_ort4);



      if k == 1
          disp(['Original simulator'])
          disp(['The original simulator at cyan state in case 1 has ', num2str(mean(FPT_ori1),'%5.1f'),'+-',num2str(std(FPT_ori1)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The original simulator at Red state in case 1 has ', num2str(mean(FPT_ori2),'%5.1f'),'+-',num2str(std(FPT_ori2)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The original simulator at cyan state in case 2 has ', num2str(mean(FPT_ori3),'%5.1f'),'+-',num2str(std(FPT_ori3)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The original simulator at Red state in case 2 has ', num2str(mean(FPT_ori4),'%5.1f'),'+-',num2str(std(FPT_ori4)*1.96/sqrt(N_IC),'%5.1f')])

          disp(['Oblique projection'])
          disp(['The ATLAS simulator at cyan state in case 1 has ', num2str(mean(FPT_sim1),'%5.1f'),'+-',num2str(std(FPT_sim1)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at Red state in case 1 has ', num2str(mean(FPT_sim2),'%5.1f'),'+-',num2str(std(FPT_sim2)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at cyan state in case 2 has ', num2str(mean(FPT_sim3),'%5.1f'),'+-',num2str(std(FPT_sim3)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at Red state in case 2 has ', num2str(mean(FPT_sim4),'%5.1f'),'+-',num2str(std(FPT_sim4)*1.96/sqrt(N_IC),'%5.1f')])

          disp(['Orthogonal Projection'])
          disp(['The ATLAS simulator at cyan state in case 1 has ', num2str(mean(FPT_ort1),'%5.1f'),'+-',num2str(std(FPT_ort1)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at Red state in case 1 has ', num2str(mean(FPT_ort2),'%5.1f'),'+-',num2str(std(FPT_ort2)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at cyan state in case 2 has ', num2str(mean(FPT_ort3),'%5.1f'),'+-',num2str(std(FPT_ort3)*1.96/sqrt(N_IC),'%5.1f')])
          disp(['The ATLAS simulator at Red state in case 2 has ', num2str(mean(FPT_ort4),'%5.1f'),'+-',num2str(std(FPT_ort4)*1.96/sqrt(N_IC),'%5.1f')])

        end

        relative_error_FPT(1, k) = (Mean_FPT(5,k)-Mean_FPT(1,k))/Mean_FPT(1,k);
        relative_error_FPT(2, k) = (Mean_FPT(6,k)-Mean_FPT(2,k))/Mean_FPT(2,k);
        relative_error_FPT(3, k) = (Mean_FPT(7,k)-Mean_FPT(3,k))/Mean_FPT(3,k);
        relative_error_FPT(4, k) = (Mean_FPT(8,k)-Mean_FPT(4,k))/Mean_FPT(4,k);

        relative_error_FPT(5, k) = (Mean_FPT(9,k)-Mean_FPT(1,k))/Mean_FPT(1,k);
        relative_error_FPT(6, k) = (Mean_FPT(10,k)-Mean_FPT(2,k))/Mean_FPT(2,k);
        relative_error_FPT(7, k) = (Mean_FPT(11,k)-Mean_FPT(3,k))/Mean_FPT(3,k);
        relative_error_FPT(8, k) = (Mean_FPT(12,k)-Mean_FPT(4,k))/Mean_FPT(4,k); 