# The zip function in Python takes two or more iterables and returns an iterator that aggregates elements from each of the iterables.
# In this example, two lists names and ages are passed to zip function to combine the corresponding elements of the two lists into tuples
names = ["Alice", "Bob"]; names_0=names[0]; names_1=names[1]

ages = [30, 25]; ages_0=ages[0]; ages_1=ages[1]

combined = zip(names, ages)

result = list(combined); result_0=result[0]; result_0_0=result[0][0]; result_0_1=result[0][1]; result_1=result[1]; result_1_0=result[1][0]; result_1_1=result[1][1]
