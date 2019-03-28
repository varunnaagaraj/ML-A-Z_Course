import os
rootDir = '/Users/Varun/Documents/Machine Learning A-Z Template Folder/Part 0 - Welcome to Machine Learning A-Z'#/Section 1 - Welcome to Machine Learning A-Z !'
os.chdir(rootDir)
for root, subdirs, files in os.walk(rootDir):
    for dirs in subdirs:
        splitdirs = dirs.split(" ")
        prefix = "Section"
        if splitdirs[0] == prefix:
            newDir = os.path.join(rootDir, dirs)
            print(newDir)
