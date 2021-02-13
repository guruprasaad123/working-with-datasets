import numpy as np
import pandas as pd
import os
# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
# for reading the csv
import csv
# for creating datasets
import h5py


# def predict(path) :
#     img = tf.keras.preprocessing.image.load_img(path,target_size=(150,150))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = img / 255.0
#     img = np.array([img])
#     pred = labels[np.argmax(model.predict(img))]
#     plt.imshow(img.reshape(150,150,3))
#     plt.title(pred)
    
def pre_process_image(path,normalize=True):

    # reading the image using path
    img = imread( path )

    if normalize == True:
        # normalize the pixel values
        img = img/255

    # resizing the image to (28,28,3)
    img = resize(img, output_shape=(28,28,3), mode='constant', anti_aliasing=True)

    # converting the type of pixel to float 32
    img = img.astype('float32')

    return img

def get_mapping(filename):
    
    obj = dict({})

    with open( filename , mode='r') as csv_file:
        
        csv_reader = csv.DictReader(csv_file)

        for i, row in enumerate(csv_reader):

            if 1 == 0:
                print('Column names are : ' , row )
            else :
                
                image = row['Image'] 
                _class = row['Class']

                obj[image] = _class
        
        return obj           

csv_path = os.path.join( 'train.csv' )

mapping = get_mapping( csv_path )

train_dir = 'train'

test_dir = 'test'

def get_list(dir,mapping=None):
    
    images = os.listdir( os.path.join( dir ) )

    img_list = []

    class_list = []

    for image in images:
        
        if mapping:
            # find the mapping and get the class
            _class = mapping[image]
            class_list.append(_class)

        # pre-process the image
        img = pre_process_image( os.path.join( dir , image ) )

        # appending the image into the list
        img_list.append(img)

    # converting the list to numpy array
    array_list = np.array(img_list)

    if mapping:
        
        unique_ones = np.unique(class_list).tolist()
        print('unique_ones' , unique_ones)
    
        class_map = list(map( lambda x : unique_ones.index(x) , class_list ))
        # print('class_map' , class_map)
        classes = np.array(class_map)

        return ( array_list , unique_ones , classes )
    
    else:
        return array_list

def create_dataset( obj=dict({}) , name='data' ):

    filename = '{}.h5'.format(name)

    hf = h5py.File( filename , 'w' )

    for key in obj:
            hf.create_dataset( key , data=obj[key] )
    
    hf.close()

# ['Airplane', 'Candle', 'Christmas_Tree', 'Jacket', 'Miscellaneous', 'Snowman']

train_x , list_class , train_y = get_list( train_dir , mapping )

test_x = get_list( test_dir )

train_obj = dict( { 'train_x' : train_x , 'list_class' : np.array(list_class,dtype='S') , 'train_y' : train_y } )

create_dataset( train_obj , 'train' )

print( 'train_x => ',train_x.shape )

print( 'train_y => ',train_y.shape )

print( 'test => ',test_x.shape )

test_obj = dict({ 'test_x' : test_x , 'list_class' : np.array(list_class,dtype='S') })

create_dataset( test_obj , 'test' )
