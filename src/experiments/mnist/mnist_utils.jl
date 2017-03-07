# mnist_utils
#
# A set of utility functions for grabbing the MNIST dataset.

using MNIST

# This function does two things:
# 1) Filters the X and y data to only be those in labels
# 2) Does PCA to reduce the dimensionality of X to numdim.
#
# @returns - (X_train_reduced, y_train, X_test_reduced, y_test)
function mnist_reduce_dimension_pca(L::Int, labels::Array{Int})
    # setup training set
    Xtrain, ytrain = traindata()
    # subset out 7 and 9 only
    idx = find(x -> x in labels, ytrain)
    Xtrain = Xtrain[:,idx]'
    ytrain = ytrain[idx]
    # do SVD on Xtrain
    Xmean = mean(Xtrain, 1)
    Xtrain = Xtrain .- Xmean
    Utrain, Strain, Vtrain = svd(Xtrain)

    # prepare test data
    Xtest, ytest = testdata()
    idx = find(x -> x in labels, ytest)
    Xtest = Xtest[:,idx]'
    # make sure we use training data to do renormalization
    Xtest = Xtest .- Xmean
    Xtest = Xtest * Vtrain[:,1:L] * diagm(1./Strain[1:L])

    (Utrain[:,1:L], ytrain, Xtest, ytest)
end

# this function returns the eigenfaces from the PCA decomp
# on the training data
function mnist_eigenfaces(L::Int, labels::Array{Int})
    # setup training set
    Xtrain, ytrain = traindata()
    # subset out 7 and 9 only
    idx = find(x -> x in labels, ytrain)
    Xtrain = Xtrain[:,idx]'
    ytrain = ytrain[idx]
    # do SVD on Xtrain
    Xmean = mean(Xtrain, 1)
    Xtrain = Xtrain .- Xmean
    Utrain, Strain, Vtrain = svd(Xtrain)

    Vtrain[:,1:L]
end
