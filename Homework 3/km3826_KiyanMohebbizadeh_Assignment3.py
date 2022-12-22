# packages
import nibabel as nib
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from nilearn.maskers import NiftiMasker
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

# load data
label = sio.loadmat('label.mat')
label = label['label'] # select only the labels over the time period
label = label.ravel() # remove extra brackets for usability

test_raw = nib.load('sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii')
retest_raw = nib.load('sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii')

def single_mask(data):
    """
    MASKING:
        - standardize standardizes around the mean for easier edge detection
        - mask strategy is set for a raw image
        - detrend helps exentuate the variance and clean the signal by passing high and low pass filters
        - opening = 1 means that once the first iteration has gone through 1 more kernel is passed to remove parts of skull remaining
        - the cutoff thresholds are set a bit more aggressively than default to be specific with the mask
    """
    masker = NiftiMasker(standardize=True,
                         mask_strategy='epi',
                         detrend=True,
                         mask_args={'opening': 1, 'lower_cutoff': .25, 'upper_cutoff': .8})

    data = masker.fit_transform(data) # fit the mask on the scan

    # IF YOU WANT TO SEE THE MASK IN ACTION UNCOMMENT THESE LINES
    # report = masker.generate_report()
    # report.open_in_browser()

    return data

def single_model(data, label):
    """
    PCA:
        - we create a dummy svm model to test out the best PCA model for the data
        - n_components is the number of features left after PCA (the number of features left after
        dimensionality reduction
        - iterate through the various pca n_components and select the highest accuracy dimension
        - dimension reduction can go up to the min(features, samples) and we have more features than samples so
        we set the last value to the number of samples (rows)
        - if having no dimensionality reduction is better we skip PCA
    """
    # test PCA accuracy
    svc = SVC(random_state=19)
    score = 0
    best_n = 0
    for n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
              55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
              105, 110, 115, 120, 125, 130, 135, 140,
              145, 150, (data.shape[0])]:
        pca = PCA(n_components=n, random_state=19)
        pca_data = pca.fit_transform(data)
        svc.fit(pca_data, label)
        if svc.score(pca_data, label) >= score:
            score = svc.score(pca_data, label)
            best_n = n

    # see if no PCA is better
    svc.fit(data, label)
    if svc.score(data, label) < score:
        pca = PCA(random_state=19, n_components=best_n)
        data = pca.fit_transform(data)
    else:
        best_n = 'NO PCA'

    """
    SVM MODEL:
        - set the parameters we want tested with the data (exclude precomputed because specific dimensions required)
        - set possible C values across normal range
        - set possible gamma values across normal range
    RANDOMIZED-CV:
        - randominzed-cv works by selecting random combinations of the parameters to test with cross validation
        - select various k values we want to test for randomized-cv (how many folds in the data)
        - set the model to our SVM model
        - select the parameters that are being tested and the values to select for each (initialized above)
        - set cv to the numder of folds 'k'
        - n_jobs=-1 allows the program to use all avaliable processors to run iterations successively
        - n_iter is the number of parameter combinations tested, this is set higher than the default 10 
        for improved accuracy at the expense of efficiency
        - set verbose to 1 if you want progress updates on the model
    """

    kernel = ['linear', 'rbf', 'poly', 'sigmoid']
    C = [.001, .01, .1, 1, 10, 100, 1000]
    gamma = [.001, .01, .1, 1, 5, 10, 100, 1000]

    random_grid = {'kernel': kernel,
                   'C': C,
                   'gamma': gamma}

    best_mvs = 0
    best_k = 0
    for k in [2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        svc = SVC(random_state=19)
        svc = RandomizedSearchCV(svc, param_distributions=random_grid, cv=k, n_jobs=-1, n_iter=30, verbose=0, random_state=19)
        svc.fit(data, label)
        if svc.best_score_ > best_mvs:
            best_mvs = svc.best_score_
            best_params = svc.best_params_
            best_score = svc.score(data, label)
            best_k = k

    # print out the results of the model
    print('Best n value for PCA:', best_n)
    print('Best k for Cross-Validation:', best_k)
    print('Best Parameters:', best_params)
    print('Mean Validation Score for Best Model:', best_mvs)
    print('Score for Model:', best_score)


# run functions
print('TEST DATA:')
test = single_mask(data=test_raw)
single_model(test, label=label)

print('\nRETEST DATA:')
retest = single_mask(data=retest_raw)
single_model(retest, label=label)



