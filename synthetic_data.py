import random

def generate_math_problem():
    a = random.randint(0, 1000)
    b = random.randint(0, 1000)
    ops = [("+", a + b), ("-", a - b)]
    op, res = random.choice(ops)
    return f"Mennyi {a} {op} {b}? Az eredmény {res}."

def generate_grammar_sentence():
    subjects = ["a macska", "a kutya", "a fiú", "a lány", "az ai"]
    actions = ["szereti", "látja", "keresi", "eszi"]
    objects = ["az almát", "a csontot", "a várost", "a feladatot"]
    s = random.choice(subjects)
    a = random.choice(actions)
    o = random.choice(objects)
    return f"{s.capitalize()} {a} {o}."

def get_synthetic_batch(size=32):
    batch = []
    for _ in range(size // 2):
        batch.append(generate_math_problem())
        batch.append(generate_grammar_sentence())
    return batch
