"""
Copyright (C) 2022 Pascal Notin, University of Oxford

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from typing import Any,AnyStr,List,Dict,Tuple,Union,Optional
import numpy as np
from slingpy import AbstractDataSource

class NumpyDataSource(AbstractDataSource):
    def __init__(self, wrapped_data_source):
        self.wrapped_data_source = wrapped_data_source
        self.row_index = {}
        for item in self._get_all_row_names():
            self.row_index[item] = item

    def __len__(self) -> int:
        return len(self.wrapped_data_source)

    def __getitem__(self, index: int) -> Any:
        return self.wrapped_data_source[index]
    
    def _get_all_row_names(self) -> List[AnyStr]:
        return range(self.__len__())

    def get_shape(self) -> List[Tuple[int]]:
        return self.wrapped_data_source.shape

    def get_data(self) -> List[np.ndarray]:
        return self.wrapped_data_source
    
    def _get_data(self) -> List[np.ndarray]:
        return self.wrapped_data_source

    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        return [self.wrapped_data_source[row_name]]

    def get_row_names(self) -> List[AnyStr]:
        return range(self.__len__())

    def get_column_names(self) -> List[List[AnyStr]]:
        return range(self.wrapped_data_source.shape[-1])

    def get_column_code_lists(self) -> List[List[Dict[int, AnyStr]]]:
        return None