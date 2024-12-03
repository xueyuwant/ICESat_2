# ICESat_2
Icesat_2 ATL13 read h5 data
This Python script processes ICESat-2 ATL13 HDF5 data files, filters the data based on specified criteria, and then saves the processed data to CSV files. It includes functionalities for checking file integrity, applying spatial filters, and saving the results in an organized format.

Features
Reads ICESat-2 ATL13 HDF5 data from .h5 files.
Filters data based on specified spatial bounds (latitude, longitude).
Calculates timestamps if required.
Exports filtered data to CSV files, one for each beam of the data.
