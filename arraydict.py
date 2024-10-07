# %%
import numpy as np
from copy import deepcopy
from tabulate import tabulate


# %%
class ArrayDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__len = 0
        self.__convert_and_validate()

    def __convert_and_validate(self):

        lens = set()

        for key, value in self.items():

            if not key.isidentifier():
                raise TypeError(f'Invalid key "{key}".')

            if isinstance(value, dict):
                value = ArrayDict(value)

            elif isinstance(value, int):
                value = np.array([value])

            elif not isinstance(value, np.ndarray):
                value = np.array(value)

            super().__setitem__(key, value)
            
            lens.add(len(value))
            
        if len(lens) > 1:
            raise ValueError('All arrays must have the same length in the first dimension.')
        else:
            self.__len = lens.pop()

    def __setitem__(self, key, value):

        if not key.isidentifier():
            raise TypeError(f'Invalid key "{key}".')
        
        if isinstance(value, dict):
            value = ArrayDict(value)

        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        
        if len(value) != len(self):
            raise ValueError(f'Expected length {len(self)}, got {len(value)}.')
        
        super().__setitem__(key, value)

    def __getitem__(self, loc):

        if isinstance(loc, int):
            loc = slice(loc, loc+1)

        if isinstance(loc, (slice, np.ndarray)):
            return ArrayDict({k: v[loc] for k, v in self.items()})
        
        elif isinstance(loc, str):
            return super().__getitem__(loc)
        
        elif isinstance(loc, list):
            if all(type(l) == int for l in loc):
                return self[np.array(loc)]
            if all(type(l) == str or type(l) == tuple for l in loc):
                d = {}
                for l in loc:
                    if type(l) == str:
                        d[l] = self[l]
                    elif type(l) == tuple:
                        if len(l) > 1:
                            d[l[0]] = self[l[0]][[l[1:]]]
                        else:
                            d[l[0]] = self[l[0]]
                return ArrayDict(d)
            else:
                raise TypeError('Invalid loc.')
        
        elif isinstance(loc, tuple):

            if len({type(l) for l in loc}) > 2:
                raise TypeError('Invalid loc.')
            
            if len(loc) > 1:
                return self[loc[0]][loc[1:]]
            else:
                return self[loc[0]]
            
        else:
            raise TypeError('Invalid loc.')  

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f'Attribute "{key}" does not exist.')
    
    def __setattr__(self, key, value):
        if key.startswith(f'_{self.__class__.__name__}__'):
            super().__setattr__(key, value)
        else:
            self[key] = value
        
    def __len__(self):
        return self.__len
        
    def __str__(self):
        return tabulate(flatten_dict(self, sep='.'), headers='keys', tablefmt='plain')
    
    def flatten(self, sep='_'):
        s = self.copy()
        self.clear()
        self.update(flatten_dict(s, parent_key=None, sep=sep))
        return self
    
    # def copy(self):
    #     return deepcopy(self)

def flatten_dict(nested_dict, parent_key=None, sep='_'):
    items = []
    for key, value in nested_dict.items():
        new_key = f'{parent_key}{sep}{key}' if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


data = {
    'outer1': {
        'col1': {'a': [1, 2, 3], 'b': [1, 2, 3]},
        'col2': [4, 5, 6]
    },
    'outer2': {
        'col3': [7, 8, 9],
        'col4': [10, 11, 12]
    },
    'outer3': [10, 11, 12]
}
ad = ArrayDict(data)
ad.flatten()

# %%
