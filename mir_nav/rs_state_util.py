from collections import OrderedDict

class RobotServerStateManager():
    def __init__(self, state_len_dict):
        '''
        Requirement `state_len_dict` as orderd dict
        '''
        self.rs_state_len_dict = OrderedDict(state_len_dict)
        self.rs_state_slice_dict = self.create_slice_dict(self.rs_state_len_dict)

    def apply_element_state(self, key, length):
        self.rs_state_len_dict[key] = length
        self.rs_state_slice_dict = self.create_slice_dict(self.rs_state_len_dict)

    def get_len(self, key):
        assert key in self.rs_state_len_dict.keys(), f'Not registered "{key}". Valid keys for len_dict are {self.rs_state_len_dict.keys()}.'
        return self.rs_state_len_dict[key]

    def get_slice(self, key):
        assert key in self.rs_state_slice_dict.keys(), f'Not registered "{key}". Valid keys for slice_dict are {self.rs_state_slice_dict.keys()}.'
        return self.rs_state_slice_dict[key]

    def get_total_len(self):
        '''
        Returns the total length of state excluding variable length.
        '''
        return sum([x for x in self.rs_state_len_dict.values() if x > 0])

    @staticmethod
    def create_slice_dict(len_dict):
        assert len([x for x in len_dict.values() if x <= 0]) <= 1
        
        start = 0
        slice_dict = OrderedDict()
        for key in len_dict:
            end = start+len_dict[key] if len_dict[key] > 0 else None
            slice_dict[key] = slice(start, end)
            start += len_dict[key]
            
        return slice_dict

    def get_from_rs_state(self, rs_state, key):
        assert key in self.rs_state_slice_dict.keys()
        return rs_state[self.rs_state_slice_dict[key]]
