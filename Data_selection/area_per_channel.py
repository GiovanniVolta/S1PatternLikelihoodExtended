import hax
import numpy as np
import pandas as pd
from hax.minitrees import TreeMaker
from collections import defaultdict
from hax.runs import get_run_info
from pax.PatternFitter import PatternFitter
from pax.configuration import load_configuration
from pax import utils, configuration, datastructure, units
from pax import exceptions
from pax.plugins.interaction_processing.S1AreaFractionTopProbability import s1_area_fraction_top_probability
from hax.corrections_handler import CorrectionsHandler
pax_config = configuration.load_configuration('XENON1T')
from pax.PatternFitter import PatternFitter


class AreaPerChannel(TreeMaker):
    """
    Restituisce extra branches
    """
    #print('AreaPerChannel')
    __version__ = '1.3'
    extra_branches = ['peaks.area_per_channel[260]',
                      'peaks.hits_per_channel[260]']
    
    def __init__(self):
        hax.minitrees.TreeMaker.__init__(self)
        hax.minitrees.TreeMaker.uses_arrays = True
    
    def get_data(self, dataset, event_list=None):
        data, _ = hax.minitrees.load_single_dataset(dataset, ['Fundamentals'], bypass_blinding=True)
        #print('data: \n', data)
        self.indices = list(data.event_number.values)
        #print('get data: \n', hax.minitrees.TreeMaker.get_data(self, dataset, event_list))
        return hax.minitrees.TreeMaker.get_data(self, dataset, event_list)

    def extract_data(self, event):
        
        event_data = {
            "s1_area_per_channel": None,
            "s1_hits_per_channel": None
        }
        
        if not len(event.interactions):
            return event_data
        
        event_num = event.event_number
        
        try:
            event_index = self.indices.index(event_num)
        except Exception:
            return event_data
        
        #print(event_num)
        
        s1 = event.peaks[event.interactions[0].s1]
        s2 = event.peaks[event.interactions[0].s2]
        
        apc = np.array(list(s1.area_per_channel))
        hpc = np.array(list(s1.hits_per_channel))
        print('s1_area_per_channel: ', type(apc))
        print('s1_hits_per_channel: ', type(hpc))
        
        try:
            event_data['s1_area_per_channel'] = apc
            print(event_data['s1_area_per_channel'])
            event_data['s1_hits_per_channel'] = hpc
            print(event_data['s1_hits_per_channel'])
        
        except exceptions.CoordinateOutOfRangeException:
            # pax does this too. happens when event out of TPC (usually z)
            return event_data
        
        #event_data.to_pickle('/home/gvolta/minitrees_test/AreaPerChannel.pkl')
        
        return event_data