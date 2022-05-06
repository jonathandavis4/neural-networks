from random import shuffle

training_data_point_count = input('Number of training data points: ')
test_data_point_count = input('Number of test data points: ')

training_data_point_count = int(training_data_point_count)
test_data_point_count = int(test_data_point_count)

data = []
with open('all_data.txt', 'r') as f:
    for line in f:
        data.append(line.split(','))
shuffle(data)

point_count = 0
with open('training_data.txt', 'w') as f:
    for point in data:
        f.write(','.join(point))
        point_count += 1
        if point_count == training_data_point_count:
            break

shuffle(data)

point_count = 0
with open('testing_data.txt', 'w') as f:
    for point in data:
        f.write(','.join(point))
        point_count += 1
        if point_count == test_data_point_count:
            break