from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def get_association_rules(df):
    freq_items = apriori(df, min_support=0.05, use_colnames=True, max_len=3)
    rules = association_rules(freq_items)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
    return rules

