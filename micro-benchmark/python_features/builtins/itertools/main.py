# This code imports the itertools module in Python, which provides various functions that help with efficient looping and iteration

import itertools

data = [
    {"name": "Alice", "city": "New York"},
    {"name": "Bob", "city": "San Francisco"},
]; data_0=data[0]; data_0_name=data[0]['name']; data_0_city=data[0]['city']; data_1=data[1]; data_1_name=data[1]['name']; data_1_city=data[1]['city']


sorted_data = sorted(data, key=lambda x: x["city"])

grouped_data = itertools.groupby(sorted_data, key=lambda x: x["city"])

for city, group in grouped_data:
    print(city, list(group))

counter = itertools.count(start=1, step=2)

# cycle() example
cycler = itertools.cycle("ABC")

# repeat() example
repeater = itertools.repeat("hello", 3)

# chain() example
chained = itertools.chain("ABC", "DEF")

# compress() example
selector = [True, False]; selector_0=selector[0]; selector_1=selector[1]
compressed = itertools.compress("AB", selector)

# permutations() example
perms = itertools.permutations("ABC", 2)

# combinations() example
combs = itertools.combinations("ABC", 2)

# product() example
cartesian = itertools.product("AB", "CD")
