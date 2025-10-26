import re


def extract_math_expression(query):
    query = query.lower()
    query = re.sub(r"(what is|calculate|solve|compute|find)", "", query)
    query = query.strip()

    pattern = r"^[\d\s\+\-\*\/\.\(\)]+$"
    if re.match(pattern, query):
        return query.replace(" ", "")

    return None


def evaluate_math(expression):
    try:
        expression = expression.replace(" ", "")

        if not re.match(r"^[\d\+\-\*\/\.\(\)]+$", expression):
            return False, None

        result = eval(expression)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return True, str(result)

    except Exception as e:
        return False, None


def is_math_query(query):
    math_keywords = [
        "calculate",
        "compute",
        "solve",
        "what is",
        "how much",
        "sum of",
        "multiply",
        "divide",
    ]
    query_lower = query.lower()

    has_keyword = any(keyword in query_lower for keyword in math_keywords)

    has_numbers = bool(re.search(r"\d", query))
    has_operators = bool(re.search(r"[\+\-\*\/]", query))

    return has_keyword and has_numbers or (has_numbers and has_operators)


def handle_math_query(query):
    if not is_math_query(query):
        return False, None

    expression = extract_math_expression(query)
    if expression:
        success, result = evaluate_math(expression)
        if success:
            return True, f"the answer is {result}"

    return False, None
