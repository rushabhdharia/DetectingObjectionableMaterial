import cPickle
import cv2
from skimage import exposure
from skimage import feature
from scipy import misc
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.externals import joblib
data = []
labels = []
path = ['assets/nipples']
prediction = []


def hog_feature(im):
    im = cv2.resize(im, (250, 250))
    # im = misc.imread('assets/nipples/nipple1.jpg', flatten=True)
    (h, hog_image) = feature.hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
    hog_image = hog_image.astype("uint8")
    return h, hog_image


def prep_training_data(im_path):
    for i in im_path:
        for image in os.listdir(i):
            im = misc.imread(os.path.join(i, image), flatten=True)

            h, hog_image = hog_feature(im)
            name = i.split('/')[1]
            data.append(h)
            labels.append(name)
            cv2.imwrite("HOGImage_train.jpg", hog_image)
    return data, labels


def train(data, labels):
    model = svm.OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=5e-05, kernel='rbf',
      max_iter=-1, nu=0.03761600681140911, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    model.fit(data)
    joblib.dump(model, 'object.pkl')


def predict(image_path):
    im = misc.imread(image_path, flatten=True)
    h, hog_image = hog_feature(im)
    cv2.imwrite("HOGImage_predict.jpg", hog_image)
    model = joblib.load('object.pkl')
    prediction = model.predict(h.reshape(1, -1))
    # prediction =1
    return prediction


data,labels = prep_training_data(path)
train(data,labels)
pred = predict('nipple1.jpg')
print pred



