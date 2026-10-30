# lets  make afunction of adding two ML functions
def add_ml_functions(func1, func2):
    def combined_function(*args, **kwargs):
        result1 = func1(*args, **kwargs)
        result2 = func2(*args, **kwargs)
        return result1 + result2
    return combined_function