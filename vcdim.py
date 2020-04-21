##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file implements the computations for the VC-Dimension of our model
## It'll deal with the creation of data, labels, and minimization

import numpy as np
import matplotlib.pyplot as plt
from fidelity_minimization import fidelity_minimization
from weighted_fidelity_minimization import weighted_fidelity_minimization
from problem_gen import problem_generator, representatives;
from test_data import tester, _claim
from save_data import write_summary, read_summary, name_folder

# data generation

def vc_data (n, perm):
    '''
    Creates data for vc-dimension problem
    args.
        n (int): number of points to create
        perm (int): permutation number, 0-(2**n)-1
    rets.
        data (array): list of (point, label)
            data[i][0] (array): (x_1, x_2) coordinates
            data[i][1] (bool): label
    '''
    data = []
    Pi = np.pi
    radius = np.sqrt(2/Pi)

    # create n equidistant points on a circle
    for i in range(n):
        x_1 = radius * np.cos(2*Pi/(i+1))
        x_2 = radius * np.sin(2*Pi/(i+1))
        # implement all possible classifications of n elements
        y = int("{0:b}".format(2**n)[i])
        data.append([[x_1,x_2],y])

    return data

def minimizer_vc(chi, layers, method, name, epochs=3000, eta=0.1, seed=0):
    '''
    Adaption of big_functions.minimizer to 1-qubit vc dimension.
    Learns different classifications in vc-theory setting
    args.
        chi (str): cost function to use:
            'fidelity_chi'
            'weighted_fidelity_chi'
        layers (int): number of layers
        method (str): minimization method to use:
            'SGD'
            another valid for for function scipy.optimize.minimize
        name (str): filename for saving output
        epochs (int): number of epochs for SGD
        eta (float): learning rate for SGD
    rets.
        none
    effect.
        files created:
            summary.txt: contains useful information for the problem
            theta.txt: contains theta parameters
            alpha.txt: contains alpha parameters
            weight.txt: contains weights as flat array if they exist
    '''

    n = 2
    acc_train = 1
    while int(acc_train):
        for i in range(2**n):
            data = vc_data(n, i)
            if chi == 'fidelity_chi':
                qubits_lab = 1
                theta, alpha, reprs = problem_generator('vcdim', 1, layers,
                                                        chi, qubits_lab =
                                                        qubits_lab)
                theta, alpha, f = fidelity_minimization(theta, alpha, data,
                                                        reprs, 'n', method,
                                                        n, eta, epochs)
                acc_train = tester(theta, alpha, data, reprs, 'n', chi)
                write_summary(chi, 'vcdim', 1, 'n', layers, method,
                              name+'_'+str(n)+'_'+str(i), theta, alpha, 0, f, acc_train, acc_train,
                              seed, epochs=epochs)
                if not int(acc_train):
                    return
            elif chi == 'weighted_fidelity_chi':
                qubits_lab = 1
                theta, alpha, weight, reprs = problem_generator('vcdim', 1,
                                                                layers, chi,
                                                                qubits_lab =
                                                                qubits_lab)
                theta, alpha, weight, f = weighted_fidelity_minimization(theta,
                                                                         alpha,
                                                                         weight,
                                                                         data,
                                                                         reprs,
                                                                         'n',
                                                                         method)
                acc_train = tester(theta, alpha, data, reprs, 'n',
                                   chi, weights=weight)
                write_summary(chi, 'vcdim', 1, 'n', layers, method,
                              name+'_'+str(n)+'_'+str(i), theta, alpha, weight, f, acc_train,
                              acc_train, seed, epochs=epochs)
                if not int(acc_train):
                    return
        n += 1

def painter_vc (chi, layers, method, name, n, p, bw=False, seed=0):
    """
        Adaption of big_functions.painter to 1 qubit vc dim.
        Args.
            chi (str): cost function name, 'fidelity_chi'/'weighted_fidelity_chi'
            layers (int): number of layers
            method (str): minimization method, 'SGD'/'L-BFGS-B'
            name (str): fname for output
            n (int): number of points
            p (int): label permutation
            bw (bool): output in B&W
        Rets.
            none
        Effect.
            Create file depicting already stored data
    """
    np.random.seed(seed)
    name +='_'+str(n)+'_'+str(p)
    classes = 2
    qubits_lab = 1
    reprs = representatives(classes, qubits_lab)
    params = read_summary(chi, 'vcdim', 1, 'n', layers, method,
                          name)
    foldname = name_folder(chi, 'vcdim', 1, 'n', layers, method)

    if chi == 'fidelity_chi':
        theta, alpha = params
        sol = classify_vc(theta, alpha, reprs, chi)
    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol = classify_vc(theta, alpha, reprs, chi, weight)

    plot_vc(sol, foldname, name)


def classify_vc(theta, alpha, reprs, chi, weights=None, grain=30):
    """
        Adaption of test_data.Accuracy_test to 1 qubit vcdim.
        Args.
            theta (array): circuit parameters 1
            alpha (array): circuit parameters 2
            reprs (array): label states for different classes
            chi (str): 'fidelity_chi'/'weighted_fidelity_chi' cost function
            weights (array): circuit parameters 3
            grain (int): inverse size of grid step
        Rets.
            classification (array):
                classification[0]: data 1st component
                classificatoin[1]: data 2nd component
                classification[2]: data label
    """
    x_grid = np.linspace(-1, 1, grain)
    size = grain*grain
    classification = np.zeros(shape=(3,size))
    classification[0] = np.concatenate([x_grid for i in range(grain)])
    classification[1] = np.concatenate(
        [np.concatenate([[x] for j in range(grain)]) for x in x_grid]
    )
    classification[2] = [_claim(theta, alpha, weights, (x,y), reprs, 'n',
                                    chi) for (x,y) in
                         zip(classification[0],classification[1])]
    return classification

def plot_vc (data, foldname, filename):
    '''
        Adaption of save_data.samples_paint to 1 qubit vcdim.
        Args.
            data (array): labelled data
                data[i][0] (array): [x_1, x_2]
                data[i][1] (bool): label
            foldname (str): output foldername
            filename (str): output filename
        Rets.
            None
        Effect.
            Creates plots with classification
    '''

    plt.figure()
    plt.scatter(data[0],data[1],s=10,c=data[2],alpha=1)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(foldname+'/'+filename)
    plt.close()
