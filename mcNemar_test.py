import numpy as np

def save_mcnemar_array(save_path, predictions):
    save_path = save_path + "_mcnemar"
    np.save(save_path, predictions)

def run_mcnemar_test():
    mc_one = np.load(".\\cifar_model_one_mcnemar.npy")
    mc_two = np.load(".\\cifar_model_two_mcnemar.npy")
    mc_one_unique = 0
    mc_two_unique = 0
    both_wrong = 0
    for i in range(0, 10000):
        if mc_one[i] == 1 and mc_two[i] == 0:
            mc_one_unique += 1
        if mc_one[i] == 0 and mc_two[i] == 1:
            mc_two_unique += 1
        if mc_one[i] == 0 and mc_two[i] == 0:
            both_wrong += 1

    print("Model 1 Right, Model 2 Wrong : %i " % mc_one_unique)
    print("Model 1 Wrong, Model 2 Right : %i " % mc_two_unique)
    print("Model 1 Wrong, Model 2 Wrong : %i " % both_wrong)
    print("Model 1 Right, Model 2 Right : %i " % (10000 - max(mc_one_unique, mc_two_unique) - both_wrong))

run_mcnemar_test()