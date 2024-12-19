import numpy as np
import matplotlib.pyplot as plt

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated

def summarize(dataset):
    summaries = []
    for attribute in zip(*dataset):
        mean = np.mean(attribute)
        stdev = np.std(attribute, ddof=1) if len(attribute) > 1 else 1e-9
        summaries.append((mean, stdev))
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    epsilon = 1e-9
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev + epsilon, 2))))
    return (1 / (np.sqrt(2 * np.pi) * (stdev + epsilon))) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    predictions = [predict(summaries, test_set[i]) for i in range(len(test_set))]
    return predictions

def get_accuracy(test_set, predictions):
    correct = np.sum(test_set[:, -1] == predictions)
    return (correct / float(len(test_set))) * 100.0

def generate_synthetic_data(num_samples, class_0_mean, class_0_stdev, class_1_mean, class_1_stdev):
    class_0 = np.random.normal(class_0_mean, class_0_stdev, (num_samples, 2))
    class_1 = np.random.normal(class_1_mean, class_1_stdev, (num_samples, 2))
    class_0 = np.c_[class_0, np.zeros(num_samples)]
    class_1 = np.c_[class_1, np.ones(num_samples)]
    dataset = np.vstack((class_0, class_1))
    np.random.shuffle(dataset)
    return dataset

def generate_synthetic_data(num_samples, class_0_mean, class_0_stdev, class_1_mean, class_1_stdev):
    class_0 = np.random.normal(class_0_mean, class_0_stdev, (num_samples, 2))
    class_1 = np.random.normal(class_1_mean, class_1_stdev, (num_samples, 2))
    class_0 = np.c_[class_0, np.zeros(num_samples)]
    print(class_0)
    class_1 = np.c_[class_1, np.ones(num_samples)]
    dataset = np.vstack((class_0, class_1))
    np.random.shuffle(dataset)
    
    return dataset


num_samples = 500
class_0_mean = [2, 2]
class_0_stdev = [1, 1]
class_1_mean = [5, 5]
class_1_stdev = [1, 1]
dataset = generate_synthetic_data(num_samples, class_0_mean, class_0_stdev, class_1_mean, class_1_stdev)

x_train= dataset[:int(0.7 * len(dataset))]
y_test = dataset[int(0.7 * len(dataset)):]

summaries = summarize_by_class(x_train)

predictions = get_predictions(summaries, y_test)

accuracy = get_accuracy(y_test, predictions)
print(accuracy)


def plot_data(dataset, predictions):
    plt.figure(figsize=(10, 6))
    for i in range(len(dataset)):
        x, y, class_value = dataset[i]
        color = 'blue' if class_value == 0 else 'red'
        plt.scatter(x, y, color=color, alpha=0.5, label=f'Clase {class_value}' if i == 0 else None)
    
    for i in range(len(y_test)):
        x, y, _ = y_test[i]
        predicted_class = predictions[i]
        color = 'green' if predicted_class == 0 else 'orange'
        plt.scatter(x, y, color=color, marker='x', label=f'Predicción {predicted_class}' if i == 0 else None)
    
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title('Datos Sintéticos y Predicciones de Naive Bayes')
    plt.legend()
    plt.show()

plot_data(dataset, predictions)