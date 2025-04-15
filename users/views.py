from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    return render(request, 'users/viewdataset.html', {})

def training(request):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import seaborn as sns
    import tensorflow as tf
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing import image
    plt.style.use("ggplot")
    
    main_train_dir = settings.MEDIA_ROOT+'//'+'Train'
    main_test_dir =settings.MEDIA_ROOT+'//'+'Test'
    print(main_train_dir)
    print(main_test_dir)
    two_thousand_dir = settings.MEDIA_ROOT+'\\'+'Train//2Thousandnote'
    five_hundered_dir = settings.MEDIA_ROOT+'//'+'Train//5Hundrednote'
    two_hundered_dir =  settings.MEDIA_ROOT+'//'+'Train//2Hundrednote'
    one_hundered_dir = settings.MEDIA_ROOT+'//'+'Train//1Hundrednote'
    fifty_dir = settings.MEDIA_ROOT+'//'+'Train//Fiftynote'
    twenty_dir = settings.MEDIA_ROOT+'//'+'Train//Twentynote'
    ten_dir = settings.MEDIA_ROOT+'//'+'Train//Tennote'
    two_thousand_names = os.listdir(two_thousand_dir)
    five_hundered_names = os.listdir(five_hundered_dir)
    two_hundered_names = os.listdir(two_hundered_dir)
    one_hundered_names = os.listdir(one_hundered_dir)
    fifty_names = os.listdir(fifty_dir)
    twenty_names = os.listdir(twenty_dir)
    ten_names = os.listdir(ten_dir)


    print(two_thousand_names[:10])
    print(five_hundered_names[:10])
    print(two_hundered_names[:10])
    print(one_hundered_names[:10])
    print(fifty_names[:10])
    print(twenty_names[:10])
    print(ten_names[:10])

    print(f"total training of 2Thousand Notes : {len(two_thousand_names)}")
    print(f"total training of 5Hundered Notes : {len(five_hundered_names)}")
    print(f"total training of 2Hundered Notes : {len(two_hundered_names)}")
    print(f"total training of 1Hundered Notes: {len(one_hundered_names)}")
    print(f"total training of 50Notes : {len(fifty_names)}")
    print(f"total training of 20Notes : {len(twenty_names)}")
    print(f"total training of 10Notes : {len(ten_names)}")

    # parameters for graph we'll output images in a 4x4
    nrows = 4
    ncols = 4

    # Index for iterating over images
    pic_index = 0

    # set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols * 2, nrows * 2)

    pic_index += 8

    two_thousand_pix = [os.path.join(two_thousand_dir, fname) 
                    for fname in two_thousand_names[pic_index-8:pic_index]]

    five_hundered_pix = [os.path.join(five_hundered_dir, fname) 
                    for fname in five_hundered_names[pic_index-8:pic_index]]


    for i, img in enumerate(two_thousand_pix + five_hundered_pix):
        sub_plot = plt.subplot(nrows, ncols, i + 1)
        sub_plot.axis("Off")
        img_read = mpimg.imread(img)
        plt.imshow(img_read)
        
    plt.show()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(7, activation="softmax")
    ])

    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(main_train_dir,
                                                    batch_size=64,
                                                    target_size=(150,150),
                                                    class_mode="categorical")

    validation_generator = validation_datagen.flow_from_directory(main_test_dir,
                                                                batch_size=16,
                                                                target_size=(150,150),
                                                                class_mode="categorical")
    
    history = model.fit(train_generator,
                        epochs=10,
                        steps_per_epoch=len(train_generator),
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator))

    model.save("model.h5")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    final_accuracy = acc[-1]  # Take the last accuracy value

    print(f"Final Training Accuracy: {final_accuracy}")

    epochs = range(len(acc))

    plt.plot(epochs, acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()

    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return render(request, "users/training.html", {"acc": acc, "loss": loss})

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

def UserPredictions(request):
    import os
    from django.conf import settings
    if request.method == 'POST':
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/test_data")
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/test_data" + filename
        path = os.path.join(settings.MEDIA_ROOT, 'test_data', filename)

        # Load the model
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.h5')
        model = load_model(model_path, compile=False)

        # Load and preprocess the image
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Apply necessary preprocessing for the chosen model

        # Make predictions
        probabilities = model.predict(x)
        predicted_class = np.argmax(probabilities, axis=1)[0]

        # Map predicted class to a label
        labels = ["One Hundred Rupees", "Two Hundred Rupees", "Two Thousand Rupees", "Five Hundred Rupees",
                  "Fifty Rupees", "Ten Rupees", "Twenty Rupees"]
        result = labels[predicted_class]

        # Display the image
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.show()

        return render(request, "users/UploadForm.html", {'path': uploaded_file_url, 'result': result})
    else:
        return render(request, "users/UploadForm.html", {})