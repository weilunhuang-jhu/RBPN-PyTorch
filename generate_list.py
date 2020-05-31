import os
import glob

directory = 'LumoImg/lesions'
folders = os.listdir(directory)
folders.sort()

# remove txt files
txt_files = glob.glob(directory+'/*.txt')
print(txt_files)
for file in txt_files:
    os.remove(file)

# generate txt for every sub folder
for folder in folders:
    f_path = os.path.join(directory, folder)
    f = open(f_path + ".txt", "w")
    fnames = os.listdir(f_path)
    fnames.sort()
    for fname in fnames:
        fname = os.path.join(folder,fname)
        f.write(fname + "\n")
    f.close()
