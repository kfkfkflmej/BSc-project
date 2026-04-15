import re

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False


def _normalize(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    # remove common noise
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("=", " ")

    # keep only relevant characters (numbers, operators, dots, minus)
    text = re.sub(r"[^0-9\.\-\+\*\/\(\)\s]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_number(text):
    """
    Try to extract the final number from a string.
    GSM8K-style outputs usually end with a number.
    """
    matches = re.findall(r"-?\d+\.?\d*", text)
    return matches[-1] if matches else None


def _try_numeric_eq(a, b):
    try:
        return float(a) == float(b)
    except Exception:
        return False


def _try_sympy_eq(a, b):
    if not SYMPY_AVAILABLE:
        return False

    try:
        return sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0
    except Exception:
        return False


def is_equiv(model_pred, ground_truth, verbose=False):
    """
    Heuristic math equivalence checker used in GSM8K-style evaluation.
    """

    pred_raw = _normalize(model_pred)
    gt_raw = _normalize(ground_truth)

    pred_num = _extract_number(pred_raw)
    gt_num = _extract_number(gt_raw)

    # 1. numeric comparison (most important for GSM8K)
    if pred_num is not None and gt_num is not None:
        ok = _try_numeric_eq(pred_num, gt_num)

        if verbose:
            print(f"[math_equiv] numeric pred={pred_num} gt={gt_num} -> {ok}")

        return ok

    # 2. symbolic equivalence fallback (if expressions exist)
    if "=" in pred_raw or "=" in gt_raw:
        ok = _try_sympy_eq(pred_raw, gt_raw)

        if verbose:
            print(f"[math_equiv] sympy pred={pred_raw} gt={gt_raw} -> {ok}")

        return ok

    # 3. fallback exact-ish string match
    ok = pred_raw == gt_raw

    if verbose:
        print(f"[math_equiv] string pred={pred_raw} gt={gt_raw} -> {ok}")

    return ok