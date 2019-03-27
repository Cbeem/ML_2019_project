from NNlib import *

# r = random.randint(10001, 49999)
# print(training_data[1][r])
# im = training_data[0][r]
# plt.imshow(im)
# plt.show()



def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]
    return data, labels


# training_data[n:] split AFTER the first n items
# training_data[:n] split THE first n items

# _training_data, _training_labels = loadMNIST("train", "MNISTNumbers")
#
# _evaluation_data = _training_data[:10000]
# _evaluation_labels = _training_labels[:10000]
# evaluation_data = [_evaluation_data, _evaluation_labels]
# standardised_evaluation = [[None for _ in range(len(_evaluation_data))] for _ in range(2)]

# _training_data = _training_data[10000:]
# _training_labels = _training_labels[10000:]
# training_data = [_training_data, _training_labels]
# standardised_training = [[None for _ in range(len(_training_data))] for _ in range(2)]

training_data = loadMNIST("train", "MNISTNumbers")
standardised_training = [[None for _ in range(len(training_data[0]))] for _ in range(2)]

test_data = loadMNIST("t10k", "MNISTNumbers")
standardised_test = [[None for _ in range(len(test_data[0]))] for _ in range(2)]

# ========================================= #
#             Data preparation              #
# ========================================= #

def standardise_image(image):
    """ The images of the MNIST dataset are provided as two dimensional arrays, with greyscale ranging from 0 to 255.
        This function standardises each image; setting the mean to 0 and the variance to 1."""
    raveled = np.asarray(image).ravel()
    return (raveled - raveled.mean()) / raveled.std()


# Standardise the training data
for index, image in enumerate(training_data[0]):
    standardised_training[0][index] = standardise_image(image)
    standardised_training[1][index] = training_data[1][index]

# # Standardise the evaluation data
# for index, image in enumerate(evaluation_data[0]):
#     standardised_evaluation[0][index] = standardise_image(image)
#     standardised_evaluation[1][index] = evaluation_data[1][index]

# Standardise the test data
for index, image in enumerate(test_data[0]):
    standardised_test[0][index] = standardise_image(image)
    standardised_test[1][index] = test_data[1][index]


# ========================================= #
#       network definition/creation         #
# ========================================= #

# net = NeuralNet()
#
# net.add_input_layer(784)
# net.add_hidden_layer(200, activation_function="relu", learning_rate=0.001)
# net.add_output_layer(10, learning_rate=0.001)
#
# print("Network created")
#
# net.save_network_to_disk("Project_76_untrained", overwrite=False)
#
# # ========================================= #
# #                 Training                  #
# # ========================================= #
#
# print("Training network")
# for epoch in range(10):
#     print("Epoch {}".format(epoch + 1))
#     for img_index, image in enumerate(standardised_training[0]):
#         img_number = standardised_training[1][img_index]
#         target = [1 if img_number == i else 0 for i in range(10)]
#         net.forward_pass(image)
#         net.backward_pass(target)
# print("Training finished")


# ========================================= #
#                 Testing                   #
# ========================================= #
net = NeuralNet.load_network_from_disk("Project_76_trained")

print("Testing network")
num_correct = 0
incorrect_images = []
for img_index, image in enumerate(standardised_test[0]):
    img_number = standardised_test[1][img_index]
    target = [1 if img_number == i else 0 for i in range(10)]
    r = net.forward_pass(image)

    if img_number == r:
        num_correct += 1
    else:
        incorrect_images.append([img_index, r])

# net.save_network_to_disk("Project_76_trained", overwrite=False)

percent_correct = (num_correct / len(standardised_test[0])) * 100
print("{}% correct!. {} out of {}".format(percent_correct, num_correct, len(standardised_test[0])))

for img_index, prediction in incorrect_images:
    actual = test_data[1][img_index]
    print("Prediction: {} - Actual: {}".format(prediction, actual))
    im = test_data[0][img_index]
    plt.imshow(im)
    plt.show()

# # ============================================================
# #           Code for hyperparameter selection
# # ============================================================
#
# def evaluate_combination(num_of_nodes, act_fun):
#     NUM_OF_EPOCHS = 10
#
#     # Create new network for every combination
#     net = NeuralNet()
#     net.add_input_layer(784)
#     net.add_hidden_layer(num_of_nodes, activation_function=act_fun, learning_rate=0.001)
#     net.add_output_layer(10, learning_rate=0.001)
#
#     for _ in range(NUM_OF_EPOCHS):
#         for img_index, img in enumerate(standardised_training[0]):
#             img_number = standardised_training[1][img_index]
#             target = [1 if img_number == i else 0 for i in range(10)]
#             net.forward_pass(img)
#             net.backward_pass(target)
#
#     total_eval_loss = 0
#     total_correct = 0
#     for img_index, img in enumerate(standardised_evaluation[0]):
#         img_number = standardised_evaluation[1][img_index]
#         target = [1 if img_number == i else 0 for i in range(10)]
#         r, _loss = net.forward_pass_loss(img, target)
#         total_eval_loss += _loss
#         if r == img_number:
#             total_correct += 1
#
#     return total_eval_loss, total_correct
#
#
# act_functions = ["sigmoid", "relu", "tanh"]
# all_num_of_nodes = [100, 200, 400]
#
# for num_of_nodes in all_num_of_nodes:
#     for act_fun in act_functions:
#         loss, num_correct = evaluate_combination(num_of_nodes, act_fun)
#         label = act_fun + " " + str(num_of_nodes)
#         print("Error: %.4f" % round(loss, 4), "Number correct: {}".format(num_correct), label)
