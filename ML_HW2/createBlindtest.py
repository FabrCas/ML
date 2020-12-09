import os, random
import shutil
dataset = "Dataset"

# choose randomly the samples
random.seed(42)
for a in range(100):
    blindTest = "Blindtest"
    class_name = random.choice(os.listdir(dataset))
    class_path= os.path.join(os.path.normpath(dataset), class_name)
    image = random.choice(os.listdir(class_path))
    path = os.path.join(os.path.normpath(class_path), image)
    print(path)
    # copy the sample in the Blindtest directory
    blindTest = os.path.join(os.path.normpath(blindTest), class_name)
    shutil.copy(path, blindTest)


