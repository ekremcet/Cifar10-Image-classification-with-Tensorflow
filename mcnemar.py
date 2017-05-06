# Ekrem Çetinkaya S004228
# McNemar test for 2 CNN

import numpy as np

def save_mcnemar_array(save_path, predictions):
    save_path = save_path + "_mcnemar"
    np.save(save_path, predictions)

def run_mcnemar(mc_one_path, mc_two_path):
    # Load mcnemar prediction arrays
    mc_one = np.load(mc_one_path)
    mc_two = np.load(mc_two_path)
    mc_one_unique = 0
    mc_two_unique = 0
    both_wrong = 0
    # Compare predictions of two networks
    for i in range(0, 10000):
        if mc_one[i] == 1 and mc_two[i] == 0:
            mc_one_unique += 1
        if mc_one[i] == 0 and mc_two[i] == 1:
            mc_two_unique += 1
        if mc_one[i] == 0 and mc_two[i] == 0:
            both_wrong += 1
    # Print McNemar test results
    print("Model 1 Right, Model 2 Wrong : %i " % mc_one_unique)
    print("Model 1 Wrong, Model 2 Right : %i " % mc_two_unique)
    print("Model 1 Wrong, Model 2 Wrong : %i " % both_wrong)
    print("Model 1 Right, Model 2 Right : %i " % (10000 - mc_one_unique - mc_two_unique - both_wrong))