import os, random
import shutil
dataset = "Dataset"
result = ""
occurrences={}
class_names= ['Crackers', 'Dust_Cloths', 'Muesli_box', 'Pears', 'Tea_drink_bottle', 'dinner_plate', 'paper_bag', 'plastic_spoon']
# choose randomly the samples
random.seed(42)

for a in range(100):
    blindTest = "Blindtest"
    class_name = random.choice(os.listdir(dataset))
    try:
        occurrences[class_name] += 1
    except KeyError:
        occurrences[class_name] = 1
    class_path= os.path.join(os.path.normpath(dataset), class_name)
    image = random.choice(os.listdir(class_path))
    path = os.path.join(os.path.normpath(class_path), image)
    print(path)
    # copy the sample in the Blindtest directory
    blindTest = os.path.join(os.path.normpath(blindTest), class_name)
    shutil.copy(path, blindTest)



print(occurrences)

with open("results_test.txt",'a') as file_object:
    for k,v in occurrences.items():
        for n in range(v):
            file_object.write(k + "\n")