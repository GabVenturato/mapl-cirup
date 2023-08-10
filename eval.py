import pickle

if __name__ == '__main__':
    # shitty code but should let you start easily with the result pickle file
    with open("examples_learning/log.pickle", 'rb') as f:
        results = pickle.load(f)

        params, losses, params_names = results

        print(losses[-1])  # the last loss is -1 because I haven't computed yet the true one, you shouldn't need it

        for i, param in enumerate(params[-1]):
            print(f'{params_names[i]}: {param}')