import numpy as np
import pandas as pd
def S2PatternLikelihood_HE(df, path_file='/home/gvolta/Desktop/SR1/S1PatternLikelihoodExtended/S2PatternLikelihood/s2patternlikelihoodcut_he_r_phi_params_v1.txt'):
    
    s2pattern='s2_pattern_fit_top_reduced_ap'
    s2_pattern_fit='s2_pattern_fit'
        
    def powerlaw(x,amp0,power0,amp1,power1,cte):
        return amp0*x**power0+amp1*x**power1+cte   
    
    phi1=4
    phi2=10
    phi3=16
    phi4=22
    
    # Load parameters
    params_load=np.loadtxt(str(path_file))
    # Reshape parameters 
    params=[params_load[:phi1], params_load[phi1:phi1+phi2], params_load[phi1+phi2:phi1+phi2+phi3], 
            params_load[phi1+phi2+phi3:phi1+phi2+phi3+phi4]]
                     
    r_here='r_3d_nn_tf'
    phi_here='phi_3d_nn_tf'
    df[phi_here]=np.arccos(df.x_3d_nn_tf/df.r_3d_nn_tf)*np.sign(df.y_3d_nn_tf)
    df_list=[]
    R = np.linspace(0, 47, 5) 
    for i in range(len(R)-1):
        n = [phi1,phi2,phi3,phi4][i]
        td = 2*np.pi/n
        for j in range(n):

            tmin, tmax, rmin, rmax = j*td-np.pi, j*td-np.pi+td, R[i], R[i+1]
            df_box_cut=df.copy()
            box_cut=((df_box_cut[r_here]>rmin)&(df_box_cut[r_here]<rmax)&(df_box_cut[phi_here]>tmin)&(df_box_cut[phi_here]<tmax))
            df_box_cut=df_box_cut[box_cut]
            a_here=  params[i][j][0]*np.ones(len(df[box_cut]))
            b_here=  params[i][j][1]*np.ones(len(df[box_cut]))
            c_here=  params[i][j][2]*np.ones(len(df[box_cut])) 
            d_here=  params[i][j][3]*np.ones(len(df[box_cut]))
            e_here=  params[i][j][4]*np.ones(len(df[box_cut]))
            df_box_cut['CutS2PatternLikelihoodHE_a'] = a_here
            df_box_cut['CutS2PatternLikelihoodHE_b'] = b_here
            df_box_cut['CutS2PatternLikelihoodHE_c'] = c_here
            df_box_cut['CutS2PatternLikelihoodHE_d'] = d_here
            df_box_cut['CutS2PatternLikelihoodHE_e'] = e_here
            
            df_list.append(df_box_cut)
            del df_box_cut
    del df
    df_final=pd.concat(df_list)#, ignore_index=True)

    s2patternfit=df_final[s2pattern]
    s2_pattern_fit=df_final[s2_pattern_fit]
    s2=df_final['s2']
    a=df_final['CutS2PatternLikelihoodHE_a']
    b=df_final['CutS2PatternLikelihoodHE_b']
    c=df_final['CutS2PatternLikelihoodHE_c']
    d=df_final['CutS2PatternLikelihoodHE_d']
    e=df_final['CutS2PatternLikelihoodHE_e']
    s2pattern_cut_HE=((np.log10(s2patternfit)<powerlaw(np.log10(s2),a,b,c,d,e))& (s2>1e4))|((s2_pattern_fit < 0.0404*s2 + 594*s2**0.0737 - 686)& (s2<=1e4))
    df_final.loc[:,'CutS2PatternLikelihoodHE']=s2pattern_cut_HE
    return df_final
