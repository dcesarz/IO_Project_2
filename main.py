from copy import deepcopy

import classification
import preprocessing as prep
import association
import stats


def main():
    original_df = prep.load()

    sts, values = stats.get_stats(original_df)
    print(sts)
    sts.to_csv("statistics.csv")
    for val in values:
        s = '{}_values.csv'.format(val)
        values[val].to_csv(s)

    association_prepped = prep.association_prep(deepcopy(original_df))
    rules = association.get_association_rules(association_prepped)
    rules.to_csv("rules.csv")

    train_input, test_input, train_classes, test_classes = prep.classification_prep(original_df)
#, ['knn', 3], ['knn', 5], ['knn', 11], ['nbc'], ['cnn'], ['mlp'], ['svc'], ['rfc']
    methods_strings = [['dtc'], ['knn', 3], ['knn', 5], ['knn', 11], ['nbc'], ['rfc'], ['svc'], ['cnn'], ['mlp']]

    for method in methods_strings:
        result = getattr(classification, method[0])
        score = 0
        time = 0
        if method[0] == 'knn':
            score, time = result(method[1], train_input, test_input, train_classes, test_classes)
        else:
            score, time = result(train_input, test_input, train_classes, test_classes)
        method.append(score)
        method.append(time)
        print('{s} is done..'.format(s=method[0]))

    for method in methods_strings:
        if method[0] == 'knn':
            print('knn{i}_score = {s} knn{i}_time = {t}'.format(s=method[2], i=method[1], t=method[3]))
        else:
            print('{i}_score = {s} {i}_time = {t}'.format(s=method[1], i=method[0], t=method[2]))

    with open('results.txt', 'w') as file_handle:
        for list_item in methods_strings:
            file_handle.write('%s\n' % list_item)

    with open('stats.txt', 'w') as file_handle:
        for list_item in methods_strings:
            file_handle.write('%s\n' % list_item)


main()
