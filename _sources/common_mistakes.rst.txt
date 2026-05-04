#
# adding common error messages we found while running NEOST and possible solutions

#"Fortran runtime error: File already opened in another unit"
# either directory name is too long or tries to access a folder that is not empty with Resume=False?
# more details at https://github.com/JohannesBuchner/PyMultiNest/issues/171
