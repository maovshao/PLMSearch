# distutils: language = c++

from pytmalign cimport pytmalign

cdef class Pypytmalign:
    cdef pytmalign* c_pytmalign  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self):
        self.c_pytmalign = new pytmalign()

    def get_score(self, structure1, structure2):
        cdef bytes structure1_py_bytes = structure1.encode()
        cdef char* structure1_c_string = structure1_py_bytes
        cdef bytes structure2_py_bytes = structure2.encode()
        cdef char* structure2_c_string = structure2_py_bytes
        return self.c_pytmalign.get_score(structure1_c_string, structure2_c_string)

    def __dealloc__(self):
        del self.c_pytmalign

def main():
    rec_ptr = new pytmalign()  # Instantiate a pytmalign object on the heap
    
    try:
        pass
    finally:
        del rec_ptr  # delete heap allocated object

    cdef pytmalign rec_stack  # Instantiate a pytmalign object on the stack