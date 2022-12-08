cdef extern from "pytmalign.cpp":
    pass

# Declare the class with cdef
cdef extern from "pytmalign.h" namespace "python_tmalign":
    cdef cppclass pytmalign:
        pytmalign() except +
        double get_score(char * structure1, char * structure2)