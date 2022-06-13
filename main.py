import pandas as pd

if __name__ == '__main__':

    path_to_desktop = '/Users/marianagonzmed/Desktop'
    #protein_directories = ['P28845','P31645','P34913','Q99720','Q9UBN7','frames_a5a4']
    protein_directories = ['out__scores_P28845.dat']

    for dir in protein_directories:
        print(dir)
        protein_directories = os.path.join(path_to_desktop,dir)