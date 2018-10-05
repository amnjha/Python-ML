#ML Libraries
import tensorflow as tf
import numpy as np
import imread
from sklearn.linear_model import LogisticRegression
#from IPython.display import display, Image
from PIL import Image
import matplotlib.pyplot as plt
import time

#File Helper Libraries
import sys
import os
import tarfile
from urllib.request import urlretrieve
import pickle




url = 'http://commondatastorage.googleapis.com/books1000/'
percent_dl = None
data_root = "/Users/amajha/Documents/ml_data"

def dl_progress_hook(count, b_size, total_size):
    global percent_dl
    percent = int(count*b_size*100 / total_size)
    if percent_dl!=percent:
        if percent%10==0:
            sys.stdout.write("%s%%" %percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
    
    percent_dl = percent

def dl_file(filename, expected_bytes, force = False):
    dest = os.path.join(data_root, filename)
    if force or not os.path.exists(dest):
        print("Attempting to download file : ", filename)
        filename, _ = urlretrieve(url+filename, filename = dest, reporthook = dl_progress_hook)
        print("\nDownload Complete!")
    statinfo= os.stat(dest)
    if statinfo.st_size == expected_bytes:
        print("Found and Verified : ", filename)
    else:
        raise Exception(
            "Failed Verfication, expected size: "+ expected_bytes + ", Actual Size: "+ statinfo.st_size
        )
    return dest

train_filename = dl_file('notMNIST_large.tar.gz', 247336696, False)
test_filename = dl_file('notMNIST_small.tar.gz', 8458043, False)


num_classes = 10
np.random.seed(133)

def extract_files(filename, force =False):
    root =  os.path.splitext(os.path.splitext(filename)[0])[0]
    if(os.path.isdir(root) and not force):
        print("%s already present - Skipping extraction of %s" %(root, filename))
    else:
        print("Extracting data for %s. This may take a while, please wait... " %root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root,d))]
    if len(data_folders)!= num_classes:
        raise Exception("Expected %d folders, Found %d instead" %(num_classes, len(data_folders)))
    #print(data_folders)
    return data_folders

train_folders = extract_files(train_filename)
test_folders =  extract_files(test_filename)

def display_random_image(folder, num_images =1):
    image_files = os.listdir(folder)
    num_image_files = len(image_files)

    for i in range(num_images):
        rand_index = np.random.randint(num_image_files)
        #display(Image(image_files[rand_index]))
        image = Image.open(os.path.join(folder, image_files[rand_index]))
        image.show()

# Code to display Images 
#arr = np.random.permutation(len(train_folders))
#for val in arr:
#    display_random_image(train_folders[val])


image_size =28
pixel_depth =255.0

def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset  = np.ndarray(shape= (len(image_files), image_size, image_size))
    print(folder)

    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imread.imread(image_file, as_grey=True).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape!=(image_size, image_size):
                raise Exception("Unexpected Image Shape %s" %image_data.shape)
            dataset[num_images,:,:] = image_data
            num_images+=1
        except (IOError, RuntimeError) as e:
            print("Could not read file :", image_file, " : ", e, "Skipping this file.")

    dataset = dataset[0:num_images,:,:] 
    if num_images < min_num_images:
        raise Exception("Count of images less than expected: %d < %d" %(num_images, min_num_images))

    print("Full Dataset Tensor: ", dataset.shape)
    print("Mean : ", np.mean(dataset))
    print("Standard Deviation: ", np.std(dataset))
    return dataset       

def pickle_files(data_folders, min_num_images_pers_class, force =False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder+".pickle"
        dataset_names.append(set_filename)

        if os.path.exists(set_filename) and not force:
            print("Pickle File %s aleady exists, Skipping" %set_filename)
        else:
            data = load_letter(folder, min_num_images_pers_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unable to Picke files for : ", set_filename, ":", e)
    
    return dataset_names

train_datasets = pickle_files(train_folders, 45000)
test_datasets = pickle_files(test_folders, 1800)

#print(train_datasets)
#print(test_datasets)

def verify_pickle_file(pickle_file_name):
    with open(pickle_file_name, 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)
        rand_index = np.random.randint(len(dataset))

        plt.figure()
        plt.imshow(dataset[rand_index,:,:])
        plt.show()

#rand = np.random.random_integers(0, high = len(train_datasets))
#verify_pickle_file(train_datasets[rand])

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)

    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t =0,0
    end_v, end_t = vsize_per_class, tsize_per_class

    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file,'rb') as pk:
                data = pickle.load(pk)
                np.random.shuffle(data)
                if valid_dataset is not None:
                    valid_letter = data[:vsize_per_class, :,:]
                    valid_dataset[start_v:end_v,:,:] = valid_letter

                    valid_labels[start_v:end_v] = label
                    start_v+=vsize_per_class
                    end_v+=vsize_per_class

                train_letter = data[vsize_per_class:end_l,:,:]
                train_dataset[start_t:end_t,:,:] = train_letter
                train_labels[start_t:end_t] = label
                start_t+=tsize_per_class
                end_t+=tsize_per_class
        except Exception as e:
            print("Exception occured : ",e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size =10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)

_, _ , test_dataset, test_labels = merge_datasets(test_datasets, test_size)

#print('Training:', train_dataset.shape, train_labels.shape)
#print('Validation:', valid_dataset.shape, valid_labels.shape)
#print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
    permutation = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_lables = labels[permutation]
    return shuffled_dataset, shuffled_lables

train_dataset,train_labels = randomize(train_dataset, train_labels)
valid_dataset,valid_labels = randomize(valid_dataset, valid_labels)
test_dataset,test_labels = randomize(test_dataset, test_labels)

def validate_randomization():
    i = np.random.randint(len(train_dataset))
    plt.figure()
    plt.imshow(train_dataset[i])
    plt.show()

    print(train_labels[i])

#validate_randomization()

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  print("saved")
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

#statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)

def check_overlap(images_1, images_2):
    images_1.flags.writeable =False
    images_2.flags.writeable =False

    print("Start Hashing")
    im_1_hash = set([hash(image.tobytes()) for image in images_1])
    im_2_hash = set([hash(image.tobytes()) for image in images_2])

    overlap = set.intersection(im_1_hash, im_2_hash)
    return overlap

#print("Train, Valid Overlap : ", len(check_overlap(train_dataset, valid_dataset)))
#print("Test, Valid Overlap : ", len(check_overlap(test_dataset, valid_dataset)))
#print("Train, Test Overlap : ", len(check_overlap(train_dataset, test_dataset)))

pickle_file = os.path.join(data_root, 'notMNIST.pickle')
with open(pickle_file,'rb') as pk:
    pk_data =pickle.load(pk)
    train_dataset = pk_data['train_dataset']
    train_labels = pk_data['train_labels']
    test_dataset = pk_data['test_dataset']
    test_labels = pk_data['test_labels']
    print("Load Successful")

n_samples_train = 20000
n_samples_test = len(test_dataset)

train_dataset.shape = (len(train_dataset), image_size*image_size)
test_dataset.shape = (len(test_dataset), image_size*image_size)

limited_train_data = train_dataset[:n_samples_train,:]
limited_train_labels = train_labels[:n_samples_train]

limited_test_data = test_dataset[:n_samples_test,:]
limited_test_labels = test_labels[:n_samples_test]

clf = LogisticRegression(solver='saga', multi_class='multinomial', verbose=1, n_jobs=-1, max_iter=5000, random_state=145)
print("Start training!!")
start_time = time.clock()
clf.fit(limited_train_data, limited_train_labels)
time_taken = time.clock()-start_time
print("Training Complete in time: ", time_taken)
score = clf.score(limited_test_data, limited_test_labels)
print("Classifier Accuracy : ", score)