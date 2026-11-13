from task3.data import _to_float, mathvista_normalize

FF = lambda at, prec=None: {"question_type": "free_form", "answer_type": at, "precision": prec, "choices": None}
MC = lambda ch: {"question_type": "multi_choice", "answer_type": "text", "precision": None, "choices": ch}


def test_numbers():
    assert _to_float(r"7.119 \times 10^3 \, \text{s}") == 7119.0
    assert _to_float("10^8") == 1e8
    assert _to_float(r"10 \mathrm{~cm} / \mathrm{s}^2") == 10.0
    assert abs(_to_float(r"2.79 \times 10^{-1} \, \text{N/C}") - 0.279) < 1e-12
    assert _to_float("5-10 years") is None
    assert _to_float("1,000") == 1000.0
    assert _to_float(r"\frac{1}{2}") == 0.5
    assert _to_float("45^\\circ") == 45.0


def test_types():
    assert mathvista_normalize("3.0", FF("integer")) == "3"
    assert mathvista_normalize("1.25", FF("float", 1.0)) == "1.2"
    assert mathvista_normalize(r"2015 \text{ and } 2016", FF("list")) == "LIST:2015,2016"
    assert mathvista_normalize("(B)", MC(["x", "y"])) == "B"
    assert mathvista_normalize("9.45", MC(["6.00", "9.45"])) == "B"
    assert mathvista_normalize("y", MC(["x", "y"])) == "B"
    assert mathvista_normalize(None, FF("integer")) == "NO_ANSWER"
    assert mathvista_normalize("F", MC(["a", "b"])).startswith("OTHER:")


if __name__ == "__main__":
    test_numbers()
    test_types()
    print("ok")
