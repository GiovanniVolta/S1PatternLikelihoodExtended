import hax
from hax.minitrees import TreeMaker
from collections import defaultdict
from pax import units, configuration, datastructure
pax_config = configuration.load_configuration('XENON1T')
from pax.PatternFitter import PatternFitter
import numpy as np
from pax import utils

from pax.configuration import load_configuration
from pax import exceptions

class S2PatternReducedAP(TreeMaker):
    __version__ = '1.0'
    extra_branches = ['peaks.area_per_channel*']
    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
      
        # We need to pull some stuff from the pax config
        self.pax_config = load_configuration("XENON1T")
        self.channels_tot = self.pax_config['DEFAULT']['channels_in_detector']

        self.channels_top = self.pax_config['DEFAULT']['channels_top']
        self.is_pmt_alive = np.array(self.pax_config['DEFAULT']['gains']) > 1
     
        qes = np.array(self.pax_config['DEFAULT']['quantum_efficiencies'])
        
        # Create PMT array of booleans for use in likelihood calculation
        #self.is_pmt_in= np.ones(len(self.channels_top), dtype=bool)  # Default True
        self.is_pmt_in=self.is_pmt_alive[self.channels_top]
        # After Pulse Channel 
        self.apc = self.pax_config['DesaturatePulses.DesaturatePulses']['large_after_pulsing_channels']
        self.apc_top = list(set(self.apc).intersection(
            self.pax_config['DEFAULT']['channels_top']))
        self.s2_pattern_fitter = PatternFitter(
                    filename=utils.data_file_name(self.pax_config['WaveformSimulator']['s2_fitted_patterns_file']),
                    zoom_factor=self.pax_config['WaveformSimulator'].get('s2_fitted_patterns_zoom_factor', 1),
                    adjust_to_qe=qes[self.channels_top],
                    default_errors=(self.pax_config['DEFAULT']['relative_qe_error'] + 
                                   self.pax_config['DEFAULT']['relative_gain_error'])
        )
    def get_data(self, dataset, event_list=None):
        # If we do switch to new NN later get rid of this stuff and directly use those positions!
        # WARNING: This 'bypass_blinding' flag should only be used for production and never for analysis (see #211)
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Corrections', 'Fundamentals'])
        self.x = data.x_observed_nn_tf.values
        self.y = data.y_observed_nn_tf.values
        self.z = data.z_observed.values
        self.indices = list(data.event_number.values)
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)
        
    def extract_data(self, event):
        result = dict()
        # If there are no interactions cannot do anything
        if not len(event.interactions):
            return result
        interaction = event.interactions[0]
        s2 = event.peaks[interaction.s2]

        result['s2_area_from_top_ap_pmt'] = np.sum(
            np.array(list(s2.area_per_channel))[self.apc_top])
        event_num = event.event_number
        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return result
        area_per_channel = np.array(list(s2.area_per_channel))[self.channels_top]
# Pattern fit with all PMTs
        try:
            result['s2_pattern_fit_top']=self.s2_pattern_fitter.compute_gof(
                coordinates=(self.x[event_index], self.y[event_index]),
                areas_observed=area_per_channel,
                pmt_selection=self.is_pmt_in,
                statistic='likelihood_poisson')
        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return result
            
        # Remove all PMT AP channel
        self.is_pmt_in[self.apc_top] = False
        #self.is_pmt_in[self.is_pmt_alive] = True
        try:
            result['s2_pattern_fit_top_reduced_ap']=self.s2_pattern_fitter.compute_gof(
                coordinates=(self.x[event_index], self.y[event_index]),
                areas_observed=area_per_channel,
                pmt_selection=self.is_pmt_in,
                statistic='likelihood_poisson')
        except exceptions.CoordinateOutOfRangeException as _:
            # pax does this too. happens when event out of TPC (usually z)
            return result
        return result
