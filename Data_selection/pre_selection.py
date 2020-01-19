#import pandas as pd
import numpy as np
import hax
from sys import argv
import lax
from lax.lichens import sciencerun1
from lax.lichens import sciencerun0
from lax.lichens import postsr1

print('\nThis code is meant to perform the pre-selection for certain analysis. It is important to give the right input parameters: the folder where the data are stored (file_path), the name of the file (file) and the folder where the pkl will be save (folder_name). Moreover, do not be panic if you have some error on the cut since some of them needs different minitrees.\n Bkg file: /dali/lgrandi/ccapelli/haxcache/final_SR1')

###########################################################################################################

def main(*args):
    print('\nExample: \n file_path = /dali/lgrandi/giovo/pickle_haxcache/pax_v6.10.1/S1PL_data/ \n file = Rn_1_SR1_S1PL.haxcache \n folder_name = /dali/lgrandi/giovo/pickle_haxcache/pax_v6.10.1/S1PL_data/Rn_1_cut.pkl')
    #All possible variables the user could input
    parameter_dict = {}
    for user_input in argv[1:]: #Now we're going to iterate over argv[1:] (argv[0] is the program name)
        if "=" not in user_input: #Then skip this value because it doesn't have the varname=value format
            continue
        varname = user_input.split("=")[0] #Get what's left of the '='
        varvalue = user_input.split("=")[1] #Get what's right of the '='
        parameter_dict[varname] = varvalue
    return parameter_dict
dict = main()

hax.init(experiment='XENON1T',
         pax_version_policy = 'v6.10.1')

###########################################################################################################

file_path = str(dict['file_path']) 
file_ = file_path + str(dict['file']) 
data = hax.minitrees.load_cache_file(file_)

### Pre Selection ###

#### Fiducial volume: z_3d_nn_tf < -13.45 & z_3d_nn_tf > -83.45 & r_3d_nn_tf < 39.85 ###
#### FV of 1T: (-92.9 < z_3d_nn) & (z_3d_nn < -9) & (r_3d_nn < 36.94 ###

#data['CutFiducialization'] = (data['z_3d_nn_tf'] < -13.45)&(data['z_3d_nn_tf'] > -83.45)&(data['r_3d_nn_tf'] < 39.85)
data['CutFiducialization1T'] = (data['z_3d_nn_tf'] > -92.9)&(data['z_3d_nn_tf'] < -9)&(data['r_3d_nn_tf'] < 36.94)
data = hax.cuts.selection(data, data['CutFiducialization1T'] == True, desc='CutFiducialization1T')

### S2 single scatter HE: https://github.com/XENON1T/lax/blob/master/lax/lichens/postsr1.py ###

CutS2SingleScatter_HE = postsr1.S2SingleScatter_HE()
data = CutS2SingleScatter_HE.process(data)
data = hax.cuts.selection(data, data['CutS2SingleScatter_HE'] == True, desc='CutS2SingleScatter_HE')

#### DAQVeto, Flash and MuonVeto ###

CutDAQVeto = sciencerun1.DAQVeto()
CutFlash = sciencerun1.Flash()
CutMuonVeto = sciencerun1.MuonVeto()
data = CutDAQVeto.process(data)
data = CutFlash.process(data)
data = CutMuonVeto.process(data)
data = hax.cuts.selection(data, data['CutDAQVeto'] == True, desc='CutDAQVeto')
data = hax.cuts.selection(data, data['CutFlash'] == True, desc='CutFlash')
data = hax.cuts.selection(data, data['CutMuonVeto'] == True, desc='CutMuonVeto')

### S2 Threshold ###

CutS2Threshold = sciencerun1.S2Threshold()
data = CutS2Threshold.process(data)
data = hax.cuts.selection(data, data['CutS2Threshold'] == True, desc='CutS2Threshold')

### cS2AFT HE ###

CS2AFT_Extended = postsr1.CS2AreaFractionTopExtended()
data = CS2AFT_Extended.process(data)
data = hax.cuts.selection(data, data['CutCS2AreaFractionTopExtended'] == True, desc='CutCS2AreaFractionTopExtended')

### Interaction Peaks Biggest ###

CutInteractionsPeaksBiggest = sciencerun1.InteractionPeaksBiggest()
data = CutInteractionsPeaksBiggest.process(data)
data = hax.cuts.selection(data, data['CutInteractionPeaksBiggest'] == True, desc='CutInteractionPeaksBiggest')

### S2 Width ###

CutS2Width = sciencerun1.S2Width()
data = CutS2Width.process(data)
data = hax.cuts.selection(data, data['CutS2Width'] == True, desc='CutS2Width')

### S1_pattern_fit =! NaN & cS2 =! NaN ###

data['CutS1PatternFitExist'] = np.isnan(data['s1_pattern_fit'])
data['CutCS2bExist'] = np.isnan(data['cs2_bottom_nn_tf'])
data = hax.cuts.selection(data, data['CutS1PatternFitExist'] == False, desc='CutS1PatternFitExist')
data = hax.cuts.selection(data, data['CutCS2bExist'] == False, desc='CutCS2bExist')


### S1 Area Fraction Top HE ###

CutS1AreaFractionTop_he = postsr1.S1AreaFractionTop_he()
data = CutS1AreaFractionTop_he.process(data)
data = hax.cuts.selection(data, data['CutS1AreaFractionTop_he'] == True, desc='CutS1AreaFractionTop_he')

### Position Difference HE ###

CutPosDiff_HE = postsr1.PosDiff_HE()
data = CutPosDiff_HE.process(data)
data = hax.cuts.selection(data, data['CutPosDiff_HE'] == True, desc='CutPosDiff_HE')

### S2 Pattern Likelihood ###

CutS2PatternLikelihood = postsr1.S2PatternLikelihood()
data = CutS2PatternLikelihood.process(data)
data = hax.cuts.selection(data, data['CutS2PatternLikelihood'] == True, desc='CutS2PatternLikelihood')

### Cut history ###

cut_hist = data.cut_history
print(cut_hist)

################################################################################################

### Saving data ###

where_to_save = dict['folder_name']
data.to_pickle(str(where_to_save))

################################################################################################