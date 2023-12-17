# Call of a function which was yielded.


def squares():
    n = 1
    while True:
        yield n**2
        n += 1


gen = squares()

for i in range(5):
    try:
        a += next(gen)
    except NameError:
        a = next(gen)
