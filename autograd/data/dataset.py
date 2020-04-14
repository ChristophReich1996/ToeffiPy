from typing import Any


class Dataset(object):
    '''
    This class implements a super class of a dataset
    '''

    def __len__(self) -> int:
        '''
        Method returns the length of the dataset
        :return: (int) Length
        '''
        raise NotImplemented()

    def __getitem__(self, item: int) -> Any:
        '''
        Method returns an instants of the dataset by index
        :param item: (int) Index
        :return: (Any) Instance of the dataset
        '''
        raise NotImplemented
